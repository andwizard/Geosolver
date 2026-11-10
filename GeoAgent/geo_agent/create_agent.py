import base64
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    ToolMessage
)
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from typing_extensions import TypedDict, Annotated
from langchain.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from model.llm import create_llm
from geo_agent_tools.data_tools import reading_csv_summary, reading_excel_summary
from geo_agent_tools.doc_tools import read_document, create_document, read_plan_document, write_document
from geo_agent_tools.io_tools import create_directory, create_image_review_queue_file, find_file_in_working_directory, get_image_queue_count, get_next_image_from_queue, list_directory_contents, list_working_directory_files, query_files_by_role, save_file_metadata
from geo_agent_tools.gis_tools import reading_gis_metadata, reading_gis_metadata
from geo_agent_tools.code_tools import edit_code, execute_code_file, query_function_document, read_code, write_code, execute_code
import os
from .state import CodeReviewState, CodeState, ImageReviewState, ImageReviewState, ManageState, MapState, PlanState, ReportReviewState, ReportState, State
from utils.logger import get_logger_with_session
from utils.utils import _extract_message_content
import re

def create_planner_agent(
    llm: ChatOpenAI,
    tools: list[tool],
    system_prompt: str,
    state: State,
    working_directory: str = None
) -> CompiledStateGraph:
    
    def get_task_category(planState: PlanState) -> PlanState:
        logger = get_logger_with_session()
        logger.info("Processing agent: planner(TaskParser)")

        # 在工作目录创建planner文件夹
        planner_dir = os.path.join(os.environ["WORKING_DIRECTORY"], "planner")
        if not os.path.exists(planner_dir):
            os.makedirs(planner_dir)

        def _modify_planState(planState: PlanState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     """
                     You are *TaskParser*, a specialized AI assistant in a GIS TASK PLANNING team. The system also includes other agent like [*DataReader*, *TaskPlanner*].
                     Pretend you are an expert with 20 years of experience in GIS field. You need to parse the user's instructions and determine which GIS task category the user's instructions belong to. You need to think carefully about the GIS task and pay attention to the output file that the user expects.
                     There are six categories depending on the output file format of the task:
                     1. *DataAnalysis*: Returns the final_processed_data_files to the user, including vector files(shp, geojson, gpkg, etc.), raster files(tif, img, etc.), table files(csv, xlsx, etc.), txt files, without image_files and markdown_files. Attention:Generating raster files is considered as *DataAnalysis*.
                     2. *Cartography*: Returns image_files(png, jpg, etc.) to the user, such as: Creating static maps (e.g., thematic maps, base maps, topographic maps), Visualizing spatial data (e.g., heat maps, density maps, time series maps), Designing dynamic and interactive maps (e.g., web mapping applications, data visualization tools), Symbolizing spatial data and map styling (e.g., color schemes, symbols, scales).
                     3. *DataAnalysisAndCartography*: Returns both final_processed_data_files and image_files(png, jpg, etc.) to the user. Different from the *Cartography* category.
                     4. *DataAnalysisAndReport*: Returns both final_processed_data_files and markdown_files to the user.
                     5. *CartographyAndReport*: Returns both image_files and markdown_files to the user.
                     6. *Synthesis*: Returns final_processed_data_files, image_files, and markdown_files to the user.

                     Example of categorization:
                     1. User: 'Generate point features at the intersections of railways and the road features' Category: #DataAnalysis#
                     2. User: 'Composite the three bands of LandSat 5 to a RGB image using the most commonly used band composition.' Category: #DataAnalysis#. Because this task requires generate raster file(tif format).
                     3. User: 'Can you show me the histogram for the DEM pixel values?' Category: #Cartography#
                     4. User: 'From this input DEM data, I want you to generate a hillshade' Category: #Cartography#
                     5. User: 'Can you write a report on the DEM pixel values?' Category: #DataAnalysisAndReport#
                     6. User: 'Perform the following tasks: 1) Count the fast food restaurants in each county and store the result in a new field named "Count". 2) Calculate the fast food accessibility score for each county as (Count / Population) * 1,000 and store the result in a new field named "Score". 3) Create a thematic map showing the fast food accessibility score for each county.' Category: #DataAnalysisAndCartography#. Because this task requires generate vector files and png files.
                     7. Users: 'I want you to measure the length of boundary of each county in km based on this input data. Then display your result in an HTML file.' Category: #DataAnalysis#, because this task requires generate HTML file, not markdown_file.
                     8. Users: 'Conduct a hospital accessibility analysis following these steps: 1) find out the nearest hospital for each Census tract. 2) compute the distance between each tract’s centroid and its nearest hospital. 3) create a raster, indicating the distance from each tract to its nearest hospital.' Category: #DataAnalysisAndCartography#, Because this task requires generate vector file and raster file(tif format).
                     9. Users: 'Generate an obesity risk behavior index of each county in the USA by analyzing the rate of visits to unhealthy food retailers (such as convenience store, alcoholic drinking places, and limited service restaurant) and the visit rate to places that support physical activity (e.g., sports centers, parks, fitness centers). Visualize the results in a thematic map to highlight the obesity risk' Category: #DataAnalysisAndCartography#
                     10. Users: 'Propose a new Walmart store location, which has similar distances to the existing four locations, while near the roads within 300 meters.' Category: #DataAnalysisAndReport#. Because this type of siting issue is more appropriate for the use of reports to describe results.

                     Your response must begin with #(gis taskcategory)#, where (gis taskcategory) represents one of the following categories: #DataAnalysis#, #Cartography#, #DataAnalysisAndCartography#, or #Synthesis#.Ensure that there is no additional text before the # symbol. After providing the task category, explain the expected output file format, the reason for choosing the category, and the difficulty level of the task. The difficulty levels are as follows: basic, intermediate, advanced, expert. Once the category and difficulty are explained, use spatial thinking to identify and describe the essential GIS principle in the task, which should inform TaskPlanner. Make sure the response strictly follows this structure and that there is no additional text before the #(gis taskcategory)#.
                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": planState["messages"]
            }
            response = prompt.invoke(state_variables).to_messages()
            return response

        agent = create_react_agent(create_llm("PLAN_LLM"), [list_working_directory_files], state_schema=PlanState, state_modifier=_modify_planState)
        response = agent.invoke(planState)
        content = _extract_message_content(response)
        # 从content中提取类别
        category_match = re.match(r'#(\w+)#', content)
        if category_match:
            planState["task_category"] = category_match.group(1)
        logger.info(content)
        logger.info(f"Classification: {planState['task_category']}")
        ai_message = AIMessage(content=content, name="TaskParser")
        planState["messages"].append(ai_message)
        return planState

    def get_task_data(planState: PlanState) -> PlanState:
        logger = get_logger_with_session()
        logger.info("Processing agent: planner(DataReader)")

        def _modify_planState(planState: PlanState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *DataReader*, a specialized AI assistant in a GIS TASK PLANNING team. The system also includes other agent like [*TaskParser*, *TaskPlanner*].
                     Pretend you are an expert with 20 years of experience in GIS field. Based on GIS task instructions given by the user, provide a detailed data information.

                     *Your Step-by-Step Process:*
                     1. Using 'list_working_directory_files' tool to get the files uploaded by user(in 'data' folder), determine if the user's data supports the completion of the case. If the user's data does not support the completion of the case, you need to make a statement.
                     2. Use the following tools to read and summarize each type of data:
                        - **Vector or Raster Data**: Use `reading_gis_metadata` to read and summarize any GIS data files.  
                        - **Shapefiles**: Read and summarize only the `.shp` files. Ignore any associated files unless explicitly instructed.  
                        - **CSV Data**: Use `reading_csv_summary` to read and summarize CSV files.  
                        - **Excel Data**: Use `reading_excel_summary` to read and summarize Excel files.
                     3. Using the 'save_file_metadata' tool to save the information of the data uploaded by the user you read, and the role parameter is 'Data'.
                     4. Using the 'write_document' tool to write the data information in markdown format for the user's task, save the file in 'planner' folder, like this: 'planner/xxx_task_data_info.md'.
                     5. After you write the data information, use 'save_file_metadata' tool to save this markdown file to the working directory, your are one part of role 'Planner', so the role in the parameter is 'Planner'.

                     Here is the markdown format example of the data information, you need to follow this format, every subsection cannot be ignored:

                     # GIS Data Overview

                     ## Vector Data (if exists, add this section)

                     ### Data 1: `data/2019am/vector_data_1_name.shp` (the path of the vector file)

                     #### 1. Data Extent
                     | Attribute     | Value                     |
                     | ------------- | ------------------------- |
                     | **Min Longitude** | `xmin_1`                |
                     | **Min Latitude**  | `ymin_1`                |
                     | **Max Longitude** | `xmax_1`                |
                     | **Max Latitude**  | `ymax_1`                |

                     #### 2. CRS (Coordinate Reference System)

                     | Attribute | Value                         |
                     | --------- | ----------------------------- |
                     | **CRS**   | `crs_1`                   |

                     #### 3. Geometry Statistics

                     | Attribute           | Value                   |
                     | ------------------- | ----------------------- |
                     | **Centroid (X)**    | `centroid_x_1`      |
                     | **Centroid (Y)**    | `centroid_y_1`      |
                     | **Spatial Variance** | `spatial_variance_1` |
                     | **Invalid Geometries** | `invalid_geometries_count` |

                     #### 4. Attribute Table Fields

                     | Field Name        | Data Type      | Sample Values                         | Field Meaning         |
                     | ----------------- | ------------- | ------------------------------------- | --------------------- |
                     | `field_name_1` | `data_type_1` | `sample_values_1`                   | `field_meaning_1` |
                    
                     ### Data 2: `data/2019am/vector_data_2_name.shp` (the path of the vector file)
                     ...
                     ---

                     ## Raster Data (if exists, add this section)

                     ### Data 1: `data/2019am/raster_data_1_name.tif` (the path of the raster file)

                     #### 1. Data Extent

                     | Attribute     | Value                     |
                     | ------------- | ------------------------- |
                     | **Min Longitude** | `xmin_raster_1`       |
                     | **Min Latitude**  | `ymin_raster_1`       |
                     | **Max Longitude** | `xmax_raster_1`       |
                     | **Max Latitude**  | `ymax_raster_1`       |

                     #### 2. CRS (Coordinate Reference System)

                     | Attribute | Value                         |
                     | --------- | ----------------------------- |
                     | **CRS**   | `crs_raster_1`            |

                     #### 3. Band Statistics

                     | Band Number | Description  | Min Value   | Max Value   | Mean Value | Std Dev   |
                     | ----------- | -----------  | ----------- | ----------- | ---------- | --------- |
                     | `1`         | Band 1       | `min_value_band_1` | `max_value_band_1` | `mean_value_band_1` | `std_dev_band_1` |

                     #### 4. Resolution

                     | Attribute     | Value                     |
                     | ------------- | ------------------------- |
                     | **X Resolution** | `resolution_x_raster_1` |
                     | **Y Resolution** | `resolution_y_raster_1` |

                     ---

                     ## Table Data (CSV / Excel) (if exists, add this section)

                     ### Data 1: `data/2019am/table_data_1_name.csv` (the path of the table file)

                     #### 1. Data Overview
 
                     | Attribute     | Value                     |
                     | ------------- | ------------------------- |
                     | **Row Count** | `num_rows`             |
                     | **Column Count** | `num_columns`         |
                     | **Column Names** | `column_names`       |
                     | **Column Types** | `column_types`       |

                     #### 2. Sample Data

                     | Column Name   | Sample Value                | Field Meaning         |
                     | ------------- | --------------------------- | --------------------- |
                     | `col_1`   | `sample_data_col_1`      | `field_meaning_1` |
                     | `col_2`   | `sample_data_col_2`      | `field_meaning_2` |
                     | `col_3`   | `sample_data_col_3`      | `field_meaning_3` |

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": planState["messages"],
                }
            response = prompt.invoke(state_variables).to_messages()
            return response

        agent = create_react_agent(create_llm("DATA_READER_LLM"), [list_working_directory_files, reading_gis_metadata, reading_csv_summary, reading_excel_summary, write_document, save_file_metadata], state_schema=PlanState, state_modifier=_modify_planState)
        response = agent.invoke(planState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="DataReader")
        planState["messages"].append(ai_message)
        return planState

    def get_task_plan(planState: PlanState) -> PlanState:
        logger = get_logger_with_session()
        logger.info("Processing agent: planner(TaskPlanner)")

        # 获取任务类别以及允许的agent
        def get_agents_by_category(category):
            category_agents = {
                "DataAnalysis": ["Coder"],
                "Cartography": ["Cartographer"],
                "DataAnalysisAndCartography": ["Coder", "Cartographer"],
                "DataAnalysisAndReport": ["Coder", "Reporter"],
                "CartographyAndReport": ["Cartographer", "Reporter"],
                "Synthesis": ["Coder", "Cartographer", "Reporter"]
            }
            agents = category_agents.get(category, [])
            return ", ".join(agents)
        task_category = planState.get("task_category")
        if not task_category:
            logger.warning("No task category found in planState")
            allowed_agents = "Coder, Cartographer, Reporter"
        else:
            allowed_agents = get_agents_by_category(task_category)
        logger.info(f"Allowed agents: {allowed_agents}")

        # 获取Planner的文件
        planner_files_info = query_files_by_role.invoke("Planner")

        def _modify_planState(planState: PlanState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *TaskPlanner*, a specialized AI assistant in a GIS TASK PLANNING team. The system also includes other agent like [*TaskParser*, *DataReader*].
                     Pretend you are an expert with 20 years of experience in GIS field. Based on GIS task instructions given by the user, provide a detailed plan in the following requirements:
                     1. The user will come up with a general case in the GIS domain, which may be a simple case or a complex case containing multiple subtasks, you have to design the general solution to solve this general task and give a detailed solution for each subtask.
                     2. You are not authorized to use GIS software such as ArcGIS or QGIS, you can't give specific Python code or specify what Python libraries to use, you can only give a theoretical description of how to implement the solution.
                     3. You only propose theoretical detailed solutions, and do not take responsibility for executing the solution.

                     *Task Decomposition Guidelines*:
                     - Do **not** divide tasks into multiple subtasks unless each subtask addresses a clearly **distinct** and **logically independent** operation.
                     - For example, filtering data and saving results should be kept as a **single subtask**, not split into two.
                     - Avoid breaking down straightforward tasks into excessively granular subtasks. Use **judgment based on complexity**: tasks involving only simple spatial operations (e.g., attribute filtering, spatial joins) should typically be handled in **one subtask**.
                     - Limit the number of subtasks to the **minimum necessary** to solve the overall task logically and cleanly.
                    
                     *Your Step-by-Step Process:*
                     1. Use 'read_document' tool to read the gis data information from *DataReader*. (The file path is: {planner_files_info})
                     2. You must use the 'write_document' tool to write the plan in markdown format for the user's task, save the file in 'planner' folder.
                     3. After you write the plan, use 'save_file_metadata' tool to save the plan to the working directory, your are one part of role 'Planner', so the role in the parameter is 'Planner'.

                     *Agent roles introduction:*
                     1.*Coder*: Write and document efficient Python code for GIS data analysis, also be responsible for generating HTML file. You must ask *Coder* to save the results of the subtask, not just print them. Coder* returns the final_processed_data_files to the user, including vector files(shp, geojson, gpkg, etc.), raster files(tif, img, etc.), table files(csv, xlsx, etc.), txt files, without image files and markdown files.
                     2.*Cartographer*: Write Python code to create maps in the GIS field. Leave the creation of maps to *Cartographer*. *Cartographer* returns image files(png, jpg, etc.) to the user, such as: Creating static maps (e.g., thematic maps, base maps, topographic maps), Visualizing spatial data (e.g., heat maps, density maps, time series maps), Designing dynamic and interactive maps (e.g., web mapping applications, data visualization tools), Symbolizing spatial data and map styling (e.g., color schemes, symbols, scales).
                     3.*Reporter*: Draft, refine, and finalize the markdown report, integrating outputs from *Coder* and *Cartographer* and ensuring the narrative is clear and cohesive.                     
                     
                     *The following are the requirements of the solution:*
                     1. The solution should begin with a precise summary of the user's request.
                     2. The solution should be detailed and complete, with clear, logical structure to ensure easy understanding.
                     3. The solution should stay strictly within the scope of the user's needs, focusing on the requested steps and including only relevant agent roles as specified.
                     4. The solution should focus on detailed theoretical aspects such as methodologies, processes, and concepts, without including practical code or specific libraries.
                     5. The solution should not mention GIS software names (e.g., ArcGIS or QGIS).
                     6. In every step, clearly specify the agent to be used, referring only to the allowed roles provided by the user (e.g., {allowed_agents}).
                     7. The format of the solution is not required, but it should be neat, aesthetically pleasing, and well-structured.
                     8. Formulas should be correctly written in Markdown, using proper mathematical notation for clear display.
                     9. The Markdown document should focus solely on the solution, without including any unrelated content such as introduction, summary, or tangential explanations.
                     

                     Here is the markdown format example of the solution, you only need to refer to the formatting template:

                     # [Task Title]

                     ## 1. Task Overview

                     ### 1.1 Task Description
                     - **Task Description**: [Provide a general description of the task, detailing what is required to accomplish it. This could include objectives, expected outcomes, or specific actions that need to be taken].

                     ### 1.2 Data Overview
                     - **Input Data**: List the datasets provided, including their format (e.g., Shapefile, Excel), and a brief description of their contents (e.g., "Candidate_dumps.xls contains potential landfill locations"), and the path of the data (e.g., "data/2017pm/Candidate_dumps.xls").
                     - **Data Processing Requirements**: Mention any preprocessing needed (e.g., coordinate system alignment, data cleaning, table to vector, etc.).

                     ### 1.3 Task Difficulty
                     - **Difficulty**: [Assess the complexity of the task. Is it a straightforward process or does it require advanced techniques or domain-specific knowledge? You can categorize it as low, medium, or high difficulty based on factors such as data quality, tools required, and technical expertise needed.].
                     - **Number of SubTasks**: [State the number of logically necessary subtasks. 
                        - Simple or moderate tasks (low/medium difficulty) should typically be completed in **one subtask**, and should not exceed **two subtasks** under any circumstances. 
                        - Only tasks involving **multiple distinct operations** (e.g., spatial modeling, multi-layer overlays, time-series analysis) may justify more subtasks. 
                        - Avoid unnecessary splitting such as separating filtering and saving into separate subtasks.
                        ].

                     ---

                     ## 2. SubTask 1: [Sub-task Title 1]
                     - **SubTask Description**: [Describe the subtask in detail, including the specific steps and methods to be used].

                     ### 2.1 Execution Steps:
                     1. **Step 1**: [Describe the first step]. (e.g., Converting Table Data to Shapefile).
                     2. **Step 2**: [Describe the next step]. (e.g., Coordinate System and Projection Check).
                     3. **Step 3**: [Describe the next step]. (e.g., Save the result of ... The results of each subtask should be saved in some format).

                     ### 2.2 Method and Rationale:
                     - **Method**: [Describe the method used to complete the task]. (e.g., Use the "XY to Point" tool to create spatial points from Excel coordinates).
                     - **Rationale**: [Explain why this method was chosen]. (e.g., Shapefile format is required for spatial queries, and coordinate reprojection ensures consistency with other datasets).
                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": planState["messages"],
                "allowed_agents": allowed_agents
                }
            response = prompt.invoke(state_variables).to_messages()
            return response

        agent = create_react_agent(create_llm("PLAN_LLM"), [list_working_directory_files, read_document, write_document, save_file_metadata], state_schema=PlanState, state_modifier=_modify_planState)
        response = agent.invoke(planState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="TaskPlanner")
        planState["messages"].append(ai_message)
        planState["allowed_roles"] = allowed_agents
        return planState
    
    workflow = StateGraph(PlanState)
    workflow.add_node("get_task_category", get_task_category)
    workflow.add_node("get_task_data", get_task_data)
    workflow.add_node("get_task_plan", get_task_plan)

    workflow.add_edge(START, "get_task_category")
    workflow.add_edge("get_task_category", "get_task_data")
    workflow.add_edge("get_task_data", "get_task_plan")
    workflow.add_edge("get_task_plan", END)

    return workflow.compile()



def create_manager_agent(
    llm: ChatOpenAI, 
    system_prompt: str, 
    members: list[str], 
    tools: list[tool]
) -> CompiledStateGraph:
    logger = get_logger_with_session()
    options = ["FINISH"] + members

    def get_next_plan(manageState: ManageState) -> ManageState:
        logger.info("Processing agent: manager(planner)")
        plan_file_content = read_plan_document.invoke("Planner")
        def _modify_planState(planState: PlanState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system", 
                        """
                        You are *Manager*, a GIS domain task manager responsible for overseeing and coordinating a GIS data analysis project that culminates in the completion of user tasks.
                        Pretend you are an expert with 20 years of experience in GIS field. Your task is to assign the next Agent role to perform the task.

                        *Agent roles introduction:*
                        1.*Coder*: Write and document efficient Python code for GIS data analysis, also be responsible for generating HTML file. You must ask *Coder* to save the results of the subtask, not just print them. Coder* returns the final_processed_data_files to the user, including vector files(shp, geojson, gpkg, etc.), raster files(tif, img, etc.), table files(csv, xlsx, etc.), txt files, without image files and markdown files.
                        2.*Cartographer*: Write Python code to create maps in the GIS field. Leave the creation of maps to *Cartographer*. *Cartographer* returns image files(png, jpg, etc.) to the user, such as: Creating static maps (e.g., thematic maps, base maps, topographic maps), Visualizing spatial data (e.g., heat maps, density maps, time series maps), Designing dynamic and interactive maps (e.g., web mapping applications, data visualization tools), Symbolizing spatial data and map styling (e.g., color schemes, symbols, scales).
                        3.*Reporter*: Draft, refine, and finalize the markdown report, integrating outputs from *Coder* and *Cartographer* and ensuring the narrative is clear and cohesive.
                        
                        *Your Step-by-Step Process:*
                        1. When a user asks for the next action role, you want to understand the progress of the current task based on previous conversations and identify the next role to run and what it will do, based on the plan developed by *Planner*. Use the 'read_document' tool to read the plan file.
                        2. If the task is not completed, Assign specific one subtask to the appropriate agents (*Reporter*, *Coder* or *Cartographer*). You must inform the Agent of the path to the files they will process. You can only assign one subtask to a character at a time.
                        3. If all subtasks are completed, output FINISH.

                        *Notes:*
                        1. *Planner* may break down simple tasks into multiple steps, and you must combine what you think are several consecutive tasks and instruct Agent roles to execute it, merging related subtasks is conducive to improving the overall system efficiency. Agent roles are powerful assistants that are capable of accomplishing some complete series of tasks.
                        2. If a role needs to perform multiple subtasks in a row, you should give these subtasks all at once to avoid giving commands to a role several times.
                        3. *Planner*'s solution may contain multiple subtasks, and you must execute all subtasks until all subtasks are completed, without omitting subtasks.                        
                        4. If the *Coder* is responsible for processing the data for the task and the *Cartographer* is responsible for drawing the image, you need to instruct the *Coder* to generate enough data to support the *Cartographer* to ensure that the *Cartographer* is able to do its job properly.
                        5. Generating raster files(tif format) is *Coder*'s job, not *Cartographer*'s.
                        6. When assigning subtasks, inform the current subtask number to the Agent.
                        """
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                    (
                        "user",
                        "Given the conversation above, who should act next? What is the index of subtasks to be performed? Tell the role the detailed steps based on the plan from *Planner*. Don't let the Agent perform tasks outside its scope of responsibility."
                        "Select one of: {options}. Do not output FINISH if subtasks are still to be scheduled at this time."
                        "Do not give duplicate subtasks! If Reviewer gives CONTINUE, continue to the next subtask or just output FINISH!"
                        "The following is the plan from *Planner*: {plan_file_content}"
                    ),
                ]
            )
            state_variables = {
                "messages": manageState["messages"],
                "options": options,
                "plan_file_content": plan_file_content
            }
            response = prompt.invoke(state_variables).to_messages()
            return response

        agent = create_react_agent(create_llm('MANAGER_LLM'), [list_working_directory_files, read_document], state_schema=ManageState, state_modifier=_modify_planState)
        response = agent.invoke(manageState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="Manager")
        manageState["messages"].append(ai_message)
        return manageState

    # Define a Pydantic model for the output schema
    class RouteSchema(BaseModel):
        subtask_num: str = Field(description="The number of the subtask to be performed by the selected agent.")
        next: str = Field(description="Next step in the process")
        task: str = Field(description="The task to be performed by the selected agent.")

    def get_json_output(manageState: ManageState) -> ManageState:
        logger.info("Processing agent: manager (Formatter)")

        # Create ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are *Formatter*, responsible for formatting the output into JSON format."
                    "Use the `RouteSchema` tool to accurately structure the manager's response."
                    "In the 'next' field, select one of the following options: {options}."
                    "In the 'task' field, provide a comprehensive, clear, and specific task description. "
                    "The task description should include the following details:"
                    "1. The objective or goal of the task."
                    "2. The steps or process that need to be followed to complete the task."
                    "3. Any necessary context, conditions, or constraints related to the task."
                    "4. Any expected outputs or results from completing the task."
                    "The subtask number should be an integer (e.g., 1, 2, 3, etc.)."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(messages=manageState["messages"][-1], options=str(options))

        # structured_llm = create_llm('FORMATTER_LLM').with_structured_output(RouteSchema)
        structured_llm = create_llm('FORMATTER_LLM').bind_tools([RouteSchema])
        parser = JsonOutputFunctionsParser(pydantic_object=RouteSchema)

        # https://python.langchain.ac.cn/docs/how_to/structured_output/
        # https://python.langchain.ac.cn/docs/how_to/output_parser_json/#without-pydantic
        # 通过工具调用的方式，获取Json输出
        ai_prompt = prompt.invoke(manageState)
        # print(ai_prompt)
        ai_msg = structured_llm.invoke(ai_prompt)
        # print(ai_msg)
        result_dict = ai_msg.tool_calls[0]["args"]
        # print(result_dict)

        # pipeline = (
        #     prompt
        #     | structured_llm
        #     | parser
        # )
        # result_dict = pipeline.invoke(manageState)
        # 将字典转换为JSON字符串
        result_str = json.dumps(result_dict, ensure_ascii=False)
        logger.info(result_str)
        ai_message = AIMessage(content=result_str, name="Formatter")
        manageState["messages"].append(ai_message)
        return manageState

    workflow = StateGraph(ManageState)
    workflow.add_node("get_next_plan", get_next_plan)
    workflow.add_node("get_json_output", get_json_output)
    workflow.add_edge(START, "get_next_plan")
    workflow.add_edge("get_next_plan", "get_json_output")
    workflow.add_edge("get_json_output", END)
    return workflow.compile()        

def create_coder_agent(
    llm: ChatOpenAI,
    tools: list[tool],
    system_prompt: str,
    state: State,
    working_directory: str = None
) -> CompiledStateGraph:
    def read_gis_metadata(codeState: CodeState) -> CodeState:
        '''
        读取GIS数据
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: coder_agent(DataReader)")
        # 获取任务
        subtask_num = codeState.get("subtask_num", "")
        code_task = codeState.get("code_task", "")
    
        # 在工作目录创建coder文件夹
        coder_dir = os.path.join(os.environ["WORKING_DIRECTORY"], "coder")
        if not os.path.exists(coder_dir):
            os.makedirs(coder_dir)

        # 在Coder目录下创建子任务文件夹
        subtask_dir = os.path.join(coder_dir, f"subtask_{subtask_num}")
        if not os.path.exists(subtask_dir):
            os.makedirs(subtask_dir)

        # 获取当前工作目录下的文件列表
        # initial_directory_contents = list_working_directory_files({"all_files": True})

        data_files_info = query_files_by_role.invoke("Data")
        logger.info(f"Data files info: {data_files_info}")
        def _modify_codeState(codeState: CodeState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system", 
                        f"""
                        You are *DataReader*, a critical member of a multi-agent system designed for code generation. Your role is to read and analyze GIS data, summarizing its information in detail. The system also includes other agents like [*CodeWriter*, *CodeReviewer*, *CodeDebugger*, and *CodeFinisher*]. Your outputs will directly contribute to the pipeline.

                        Objective: You are only responsible for reading the data and generating data summaries to provide data support for the *CodeWriter*.

                        Your Step-by-Step Process:
                        1. Use the corrected tools to read and summarize the data.
                        2. Return the data summary and the subtask folder path to the *CodeWriter*.

                        Follow the guidelines below based on the type of data provided:
                        1. Vector or Raster Data: Use the 'reading_gis_metadata' tool to read and summarize the data.
                        2. Shapefiles: Only read the .shp file for processing. Ignore associated files unless instructed otherwise.
                        3. CSV Data: Use the 'reading_csv_summary' tool to read and summarize the contents of CSV files.
                        4. Excel Data: Use the 'reading_excel_summary' tool to read and summarize the contents of Excel files.
                        5. Error Handling: Do not call incorrect tools or operations. For unsupported file types, return a clear and concise message indicating the issue.
                        6. Provide an in-depth summary of the contents of each data file, Also you have to specify the encoding of the file, which helps *CodeWriter* a lot!

                        Behavioral Guidance:
                        1. Ensure all steps align with the described methods and tools.
                        2. Avoid any error-prone actions or misinterpretations of file types.
                        3. Collaborate effectively by ensuring the summary output is clear, actionable, and integrable for the other system agents.

                        The user's instruction are: {code_task}
                        
                        The initial data files of your working directory are: {data_files_info}.

                        """
                     ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": codeState["messages"],
                "code_task": code_task,
                "data_files_info": data_files_info
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, reading_gis_metadata, reading_csv_summary, reading_excel_summary], state_schema=CodeState, state_modifier=_modify_codeState)
        response = agent.invoke(codeState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="DataReader")

        codeState["messages"].append(ai_message)
        codeState["is_task_finish"] = "CONTINUE"
        return codeState
    
    def write_gis_code(codeState: CodeState) -> CodeState:
        '''
        生成代码文件
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: coder_agent(CodeWriter)")
        # 获取子任务编号
        subtask_num = codeState.get("subtask_num", "")
        code_task = codeState.get("code_task", "")
        # 获取Data文件信息
        data_files_info = query_files_by_role.invoke("Data")
        logger.info(f"Data files info: {data_files_info}")
        def _modify_codeState(codeState: CodeState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *CodeWriter*, an essential component of a multi-agent system built for efficient code generation and execution. This system includes agents such as *DataReader*, *CodeReviewer*, *CodeDebugger*, and *CodeFinisher*. 
                     Pretend you are an expert with 20 years of experience in GIS field. Based on GIS task instructions given by the User, Your primary responsibility is to generate accurate and effective Python code based on the data summaries provided by the *DataReader*. Your outputs must align with the overall objectives of the system, enabling seamless integration and execution by other agents.

                     Your Step by Step Process: 
                     1. You have to directly write code that runs and is correct.
                     2. You have to directly write optimal code that efficiently and correctly realizes the user's task.
                     3. Avoid algorithms with high time complexity and use geospatial optimization methods to improve operational efficiency
                     4. Use 'write_code' tool to write the code according to the above requirements in the subtask directory 'coder/subtask_{subtask_num}/'.

                     Your should follow the following coding requirements:
                     1. Accuracy: Write syntactically correct and logically sound Python code that adheres to best practices and solves the intended problem.
                     2. Data Integration: Always refer to the *DataReader* for data summaries, as many GIS operations depend on specific columns, fields, or metadata from the provided data.
                     3. File Encoding: Always specify the file encoding of the file you are writing. Use 'gpd.read_file(file_path, encoding=encoding)' instead of 'gpd.read_file(file_path)'
                     4. Modularity: Write well-structured, reusable, and maintainable functions or modules, ensuring clear separation of concerns.
                     5. Error Handling: Include robust error-handling mechanisms to gracefully handle issues like missing data, incorrect inputs, or unsupported operations.                    

                     You should also follow the following GIS analysis requirements:
                     1. Data Integrity Checks: Ensure the data is complete and free from missing or incorrect values.
                     2. Handling Empty or Invalid Geometries: Remove or fix geometries that are empty or invalid.
                     3. Spatial Relationships and Topology: Ensure spatial relationships (intersects, within, contains) are valid for the analysis.
                     4. Coordinate System and CRS Consistency: Automatically check and align coordinate systems between datasets.
                     5. Buffer and Projection Units: Handle buffer zone units and projection precision carefully (degrees vs. meters).
                     6. Data Quality and Precision Control: Maintain precision where necessary, especially when working with large datasets.
                     7. Appropriate Analysis Method Selection: Choose the right spatial analysis methods based on the problem.
                     8. Handling Large Datasets: Optimize for performance and memory when working with large datasets.
                     9. Automated Projection Transformation: Automatically handle CRS transformation when working with different spatial datasets.
                     10. Spatial Aggregation: Correctly aggregate multiple geometries into one when necessary.
                     11. Adding # -*- coding: utf-8 -*- before the code file in your code file.

                     The user's instruction are: {code_task}

                     The initial data files of your working directory are: {data_files_info}.

                    """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": codeState["messages"],
                "subtask_num": subtask_num,
                "code_task": code_task,
                "data_files_info": data_files_info
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(create_llm("CODE_WRITER_LLM"), [list_working_directory_files, write_code], state_schema=CodeState, state_modifier=_modify_codeState)
        response = agent.invoke(codeState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeWriter")

        codeState["messages"].append(ai_message)
        codeState["is_task_finish"] = "CONTINUE"
        return codeState

    def review_gis_code(codeState: CodeState) -> CodeState:
        '''
        审阅代码文件
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: coder_agent(CodeReviewer)")
        code_task = codeState.get("code_task", "")
        def _modify_codeState(codeState: CodeState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *CodeReviewer*, a meticulous and essential member of a multi-agent system for generating, reviewing, debugging, and integrating Python code for GIS operations. This system includes [*DataReader*, *CodeWriter*, *CodeDebugger*, and *CodeFinisher*], each with distinct responsibilities.
                     Pretend you are an expert with 20 years of experience in GIS field. Based on GIS task instructions given by the User, Your have two primary responsibilities:
                     1. Code Review for Correctness: Examine the code generated by *CodeWriter* to identify errors, syntax issues, and inconsistencies.
                     2. Geospatial Task Validation: Ensure the code fulfills the GIS task accurately from a geographic and spatial analysis perspective.

                     Your Step-by-Step Process:
                     Step 1: Read and Understand the Code.
                        Use the 'read_code' tool to load the code file created by CodeWriter. Analyze the code line by line to understand its purpose and functionality.
                     Step 2: Verify file path and file encoding in the code.
                        Using the 'list_working_directory_files' tool, check if the file paths in the code file are correct. If any file paths are incorrect, revise them.
                        Check if the file encoding is correct. If the file encoding is incorrect, revise it.
                        Also, check if the data output path is in the 'coder' directory. If not, revise it.
                     Step 3: Verify Function Parameters.
                        Use the 'query_function_document' tool to retrieve the documentation for all GIS functions such as sjoin, clip, and buffer. Ensure all function parameters are specified correctly and used as intended. This tool only query the function name, not the class name. (e.g. geopandas.GeoDataFrame is not a valid function name).
                        Note: Errors in retrieving function documentation are due to incorrect usage of 'query_function_document' and are not errors in the code itself.
                     Step 4: Validate Syntax.
                        Check for any syntax errors or invalid constructs. Confirm the code adheres to Python standards and best practices.
                     Step 5: GIS Analysis(Critical)
                        Validate whether the code accomplishes the GIS task as expected from a spatial analysis standpoint. Evaluate the code step-by-step from input to output. Ensure it adheres to geographic principles and produces the correct outputs. 
                     Step 6: Provide Detailed Feedback and Revisions.
                        Summarize all identified errors and provide actionable feedback for corrections. You don't need to introduce the code content in your reply.
                     
                     Behavioral Guidance:
                     1. Ensure the code output is saved in the 'coder' directory.
                     2. Verify file paths and encoding correctness.
                     3. Confirm the code meets the spatial analysis requirements of the user’s GIS task.
                     4. Evaluate the time and space complexity of the code to ensure efficiency.
                     5. Don't give specific codes in your reply!
                     
                     The user's instruction are: {code_task}

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": codeState["messages"]
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, query_function_document, read_code], state_schema=CodeState, state_modifier=_modify_codeState)
        response = agent.invoke(codeState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeReviewer")
        
        codeState["reviewer_comment"] = content
        codeState["messages"].append(ai_message)
        codeState["is_task_finish"] = "CONTINUE"
        return codeState
    
    def debug_gis_code(codeState: CodeState) -> CodeState:
        '''
        运行代码文件，并进行debug
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: coder_agent(CodeDebugger)")
        subtask_num = codeState.get("subtask_num", "")
        code_task = codeState.get("code_task", "")
        code_comment = codeState.get("reviewer_comment", "")
        def _modify_codeState(codeState: CodeState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *CodeDebugger*, a critical member of a collaborative multi-agent system for code generation and execution. The system includes [*DataReader*, *CodeWriter*, *CodeReviewer*, and *CodeFinisher*], each with defined roles. 
                     
                     Pretend you are an expert with 20 years of experience in GIS field.
                     You are **fully responsible** for ensuring the code runs successfully and produces correct outputs.
                     You must **continue debugging and executing the code on your own** until the task is successfully completed.
                     **Do not delegate** execution or debugging responsibilities to other agents, and **do not stop at suggesting fixes** — you must apply, test, and verify them iteratively.

                     Your Step-by-Step Process:
                     Step 1. Use the 'read_code' tool to read the code written by *CodeWriter*. Analyze the code thoroughly to understand its structure and functionality.
                     Step 2. Combine your insights with the feedback provided by *CodeReviewer*. Reflect on their suggestions and any issues identified to refine the code.
                     Step 3. Use the 'execute_code' tool to write and run the code file and save the code file in the subtask directory 'coder/subtask_{subtask_num}/'. If the execution fails, carefully analyze the error messages to identify and resolve the issues. Debug and revise the code as needed to ensure successful execution.
                     Step 4. **Iteratively revise and re-execute the code** until it completes the user's task **successfully and without any errors**.
                     You must handle all debugging and validation steps independently — **do not stop the loop or output suggestions** until the code runs correctly and produces the expected results.

                     *Execution and Debugging Policy*:
                     - You are the **only agent** allowed and required to execute and debug the code.
                     - When a code execution fails, you must **not stop or wait for further instructions**. You must **directly modify the code yourself**, re-execute, and repeat until it runs successfully.
                     - You must **not suggest fixes without applying them**.
                     - You must **not delegate code execution or error correction** to other agents.
                     - You are expected to **fully complete the task independently**, including saving the final output if required.

                     You should use spatial analysis methods to complete the user's task, write and run complete, compliant geospatial code.

                     The user's instruction are: {code_task}

                     The specific comments of CodeReviewer are as follows: {code_comment}

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": codeState["messages"],
                "subtask_num": subtask_num,
                "code_comment": code_comment
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(create_llm("CODE_WRITER_LLM"), [list_working_directory_files, read_code, execute_code], state_schema=CodeState, state_modifier=_modify_codeState)
        response = agent.invoke(codeState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeDebugger")

        codeState["messages"].append(ai_message)
        codeState["is_task_finish"] = "CONTINUE"
        return codeState
    
    def finish_gis_code(codeState: CodeState) -> State:
        '''
        结束代码生成
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: coder_agent(CodeFinisher)")
        coder_files_info = query_files_by_role.invoke("Coder")
        logger.info(f"Coder files info: {coder_files_info}")
        data_files_info = query_files_by_role.invoke("Data")
        logger.info(f"Data files info: {data_files_info}")
        subtask_num = codeState.get("subtask_num", "")
        def _modify_codeState(codeState: CodeState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                     You are *CodeFinisher*, the concluding member of a collaborative multi-agent system designed for generating, reviewing, debugging, and integrating Python code for GIS task. Other agents in this system include [*DataReader*, *CodeWriter*, *CodeReviewer*, and *CodeDebugger*], each playing a crucial role in the pipeline.
                     Your primary responsibility is to provide a comprehensive summary of the system's outcomes, focusing on the final Python code file and the generated data results.

                     Your Step-by-Step Process:
                     1. Use the 'read_code' tool to read the code written by *CodeWriter*, you only need to read the code generated by the CodeWriter for the current task, not the previous task's code.
                     1. Summarize the Final Code File:
                        - Provide an overview of the final Python code file, including:
                            - Its purpose and intended functionality.
                            - Key operations performed in the code, especially GIS-related tasks (e.g., spatial joins, buffers, projections).
                            - Improvements or revisions made during the debugging and review stages.
                     2. Use 'save_file_metadata' tool to save the final code file metadata. You're part of the 'Coder' role.
                     3. Summarize the Generated Data Results:
                        - List and describe all data outputs generated during the system run, including:
                            - The type of data (e.g., shapefiles, CSV, raster files).
                            - Key statistics or summaries of the data (e.g., number of records, attributes, or spatial properties).
                            - The file paths to the output data.
                        - Highlight the relevance of the outputs to the overall objective of the system.
                     4. Use 'save_file_metadata' tool to save the metadata of data results(.shp, .tif, .csv, etc.) generated by 'CodeDebugger', the role parameter is 'Data'.
                     
                     Behavioral Guidance:
                     1. Ensure your summary is precise, structured, and easy to understand.
                     2. Acknowledge the contributions of other agents in the system, reflecting their inputs and efforts in your final report.
                     3. Focus on delivering actionable insights that provide value to end-users and stakeholders.

                     Output Requirements:
                     1. A section summarizing the final Python code file.
                     2. (Optional) A section summarizing all generated data results with file paths.
                     3. A final evaluation of the system’s performance, noting any areas of improvement.
                    
                     The code files in your working directory are: {coder_files_info}.

                     The data files in your working directory are: {data_files_info}.

                     The subtask number is: {subtask_num}.

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": codeState["messages"],
                "coder_files_info": coder_files_info,
                "data_files_info": data_files_info,
                "subtask_num": subtask_num
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [read_code, save_file_metadata], state_schema=CodeState, state_modifier=_modify_codeState)
        response = agent.invoke(codeState)
        ai_message = AIMessage(content=_extract_message_content(response), name="CodeFinisher")

        codeState["messages"].append(ai_message) # 继承之前的消息记录，并将CodeFinisher的消息记录进去
        codeState["is_task_finish"] = "FINISH"
        return codeState

    workflow = StateGraph(CodeState)
    workflow.add_node("read_gis_metadata", read_gis_metadata)
    workflow.add_node("write_code", write_gis_code)
    workflow.add_node("review_code", review_gis_code)
    workflow.add_node("debug_code", debug_gis_code)
    workflow.add_node("finish_code", finish_gis_code)

    workflow.add_edge(START, "read_gis_metadata")
    workflow.add_edge("read_gis_metadata", "write_code")
    workflow.add_edge("write_code", "review_code")
    workflow.add_edge("review_code", "debug_code")
    workflow.add_edge("debug_code", "finish_code")

    return workflow.compile()

def create_cartographer_agent(
    llm: ChatOpenAI, 
    tools: list[tool], 
    system_prompt: str,
    state: State,
    working_directory: str = None
    ) -> CompiledStateGraph:
    
    def read_gis_metadata(mapState: MapState) -> MapState:
        '''
        读取GIS数据
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: cartographer_agent(DataReader)")
        # 获取任务
        map_task = mapState.get("map_task", "")

        # 在工作目录创建cartographer文件夹
        cartographer_dir = os.path.join(os.environ["WORKING_DIRECTORY"], "cartographer")
        if not os.path.exists(cartographer_dir):
            os.makedirs(cartographer_dir)

        # 获取当前工作目录下的文件列表
        # initial_directory_contents = list_working_directory_files({"all_files": True})

        # 获取当前工作目录下的文件列表
        result = list_working_directory_files({"all_files": True})
        initial_directory_contents = json.loads(result)

        # 提取目录和文件内容
        directories = initial_directory_contents.get("directories", [])
        files = initial_directory_contents.get("files", [])

        # 将提取的目录和文件传递给模板
        initial_directory_contents_str = f"Directories: {directories}, Files: {files}"
        def _modify_mapState(mapState: MapState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""
                     You are *DataReader*, a critical member of a multi-agent system designed for data visualization. Your role is to read and analyze GIS data, summarizing its information in detail. The system also includes other agents like [*CodeWriter*, *CodeReviewer*, *CodeDebugger*, and *CodeFinisher*]. Your outputs will directly contribute to the pipeline.

                     Objective: You are only responsible for reading the data and generating data summaries to provide data support for the *Cartographer*.

                     The user's instruction are: {map_task}
                     The initial contents of your working directory are: {initial_directory_contents_str}.

                     Follow the guidelines below based on the type of data provided:
                     1. Vector or Raster Data: Use the 'reading_gis_metadata' tool to read and summarize the data.
                     2. Shapefiles: Only read the .shp file for processing. Ignore associated files unless instructed otherwise.
                     3. CSV Data: Use the 'reading_csv_summary' tool to read and summarize the contents of CSV files.
                     4. Excel Data: Use the 'reading_excel_summary' tool to read and summarize the contents of Excel files.
                     5. Error Handling: Do not call incorrect tools or operations. For unsupported file types, return a clear and concise message indicating the issue.
                     6. Provide an in-depth summary of the contents of each data file, Also you have to specify the encoding of the file, which helps *CodeWriter* a lot!

                     Behavioral Guidance:
                     1. Ensure all steps align with the described methods and tools.
                     2. Avoid any error-prone actions or misinterpretations of file types.
                     3. Collaborate effectively by ensuring the summary output is clear, actionable, and integrable for the other system agents.

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": mapState["messages"]
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, reading_gis_metadata, reading_csv_summary, reading_excel_summary], state_schema=MapState, state_modifier=_modify_mapState)
        response = agent.invoke(mapState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="DataReader")

        mapState["messages"].append(ai_message)
        mapState["is_task_finish"] = "CONTINUE"
        return mapState
    
    def write_gis_code(mapState: MapState) -> MapState:
        '''
        生成代码文件
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: cartographer_agent(CodeWriter)")
        map_task = mapState.get("map_task", "")
        def _modify_mapState(mapState: MapState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *CodeWriter*, an essential component of a multi-agent system built for efficient code generation and execution. This system includes agents such as [*DataReader*, *CodeReviewer*, and *CodeFinisher*]. 
                     Your primary responsibility is to generate accurate and effective Python code based on the data summaries provided by the *DataReader*. 
                     Your outputs must align with the overall objectives of the system, enabling seamless integration and execution by other agents.

                     Your should follow the following requirements:
                     1. Accuracy: Write syntactically correct and logically sound Python code that adheres to best practices and solves the intended problem.
                     2. Data Integration: Always refer to the *DataReader* for data summaries, as many GIS operations depend on specific columns, fields, or metadata from the provided data.
                     3. File Encoding: Always specify the file encoding of the file you are writing. Use 'gpd.read_file(file_path, encoding=encoding)' instead of 'gpd.read_file(file_path)'
                     4. Modularity: Write well-structured, reusable, and maintainable functions or modules, ensuring clear separation of concerns.
                     5. Error Handling: Include robust error-handling mechanisms to gracefully handle issues like missing data, incorrect inputs, or unsupported operations.

                     Visualization Requirements:
                     1. Designing appropriate visualizations that clearly communicate data trends and patterns.
                     2. Selecting the most suitable chart types (e.g., bar charts, scatter plots, heatmaps) for different data types and analytical purposes.
                     3. Providing executable Python code (using libraries such as matplotlib, seaborn, or plotly) that generates these visualizations.
                     4. Including well-defined titles, axis labels, legends, and saving the visualizations as files.
                     5. Offering brief but clear interpretations of the visual findings.
                     6. Please render the plot in the background and save it to a file (e.g., using plt.savefig()), instead of displaying it in a window with plt.show().

                     Your Step by Step Process:
                     1. The code files you write and the image results must be saved in this folder (cartographer). You have to pay attention to the output path in the code, it should be prefixed with 'cartographer/'.
                     2. Use 'query_files_by_role' tool to query all files generated by the *Coder* role, confirm the file path in the code.
                     3. Use 'write_code' tool to write the code according to the above requirements.

                     The user's instruction are: {map_task}

                    """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": mapState["messages"]
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(create_llm("CODE_WRITER_LLM"), [list_working_directory_files, write_code, query_files_by_role], state_schema=MapState, state_modifier=_modify_mapState)
        response = agent.invoke(mapState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeWriter")

        mapState["messages"].append(ai_message)
        mapState["is_task_finish"] = "CONTINUE"
        return mapState
    
    def review_gis_code(mapState: MapState) -> MapState:
        '''
        审阅代码文件
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: cartographer_agent(CodeReviewer)")
        def _modify_mapState(mapState: MapState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                     You are *CodeReviewer*, a meticulous and essential member of a multi-agent system for generating, reviewing, debugging, and integrating Python code for GIS cartography. This system includes [*DataReader*, *CodeWriter*, *CodeDebugger*, and *CodeFinisher*], each with distinct responsibilities. 
                     Your primary role is to critically evaluate the Python code generated by CodeWriter, ensuring it is error-free, functional, and aligned with the system’s goals. You will provide actionable feedback, corrections, and, if necessary, the revised code file.

                     Your Step-by-Step Process:
                     Step 1: Read and Understand the Code.
                        Use the 'read_code' tool to load the code file created by CodeWriter. Analyze the code line by line to understand its purpose and functionality.
                     Step 2: Verify file path and file encoding in the code.
                        Using the 'list_working_directory_files' tool, check if the file paths in the code file are correct. If any file paths are incorrect, revise them.
                        Also, check if the file encoding is correct. If the file encoding is incorrect, revise it.
                     Step 3: Verify Function Parameters.
                        Use the 'query_function_document' tool to retrieve the documentation for all functions, especially GIS functions such as sjoin, clip, and buffer. Ensure all function parameters are specified correctly and used as intended. Only query the function name, not the class name. (e.g. geopandas.GeoDataFrame is not a valid function name).
                        Note: Errors in retrieving function documentation are due to incorrect usage of 'query_function_document' and are not errors in the code itself.
                     Step 4: Validate Syntax.
                        Check for any syntax errors or invalid constructs. Confirm the code adheres to Python standards and best practices.
                     Step 5: Edit the code file.
                        Use the 'edit_code' tool to edit the code file.
                     Step 6: Provide Detailed Feedback and Revisions.
                        Summarize all identified errors and provide actionable feedback for corrections. You don't need to introduce the code content in your reply.
                     
                     In your reply, you should explain the results of each step, especially the results of the code Chinese path.

                     Behavioral Guidance:
                     1. Ensure the output of the code is be saved in the 'cartographer' directory.
                     2. Ensure the file encoding and file path in the code is correct.

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": mapState["messages"]
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, query_function_document, read_code, edit_code, query_files_by_role], state_schema=MapState, state_modifier=_modify_mapState)
        response = agent.invoke(mapState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeReviewer")

        mapState["messages"].append(ai_message)
        mapState["is_task_finish"] = "CONTINUE"
        return mapState
    
    def debug_gis_code(mapState: MapState) -> MapState:
        '''
        运行代码文件，并进行debug
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: cartographer_agent(CodeDebugger)")
        map_task = mapState.get("map_task", "")
        def _modify_mapState(mapState: MapState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                     f"""
                     You are *CodeDebugger*, a critical member of a collaborative multi-agent system for code generation and execution. The system includes [*DataReader*, *CodeWriter*, *CodeReviewer*, and *CodeFinisher*], each with defined roles. 
                     
                     Your primary responsibility is to execute the code file generated by *CodeWriter*, identify and resolve any errors, and ensure the code runs successfully. Your output will determine the readiness of the code for final integration.

                     Your Step-by-Step Process:
                     1. Use the 'execute_code_file' tool to run the code file edited by *CodeReviewer*.
                        - If the code does not accomplish the user's task, go to step 2.
                     2. Use the 'execute_code' tool to write and run the code file and save the code file in the 'cartographer' directory. You have to pay attention to the output path in the code, it should be prefixed with 'cartographer/'.
                        - If the execution fails, analyze the error messages and locate the source of the problem in the code.
                        - Debug the identified errors with detailed revisions to ensure the code can run without issues.
                        - Iteratively debug and test the code until it executes successfully.
                     3. Repeat step 2 until the user's task is complete.

                     Tool Usage Guidance:
                     - 'execute_code_file': run the code file existing in the 'cartographer' directory.
                     - 'execute_code': write and run the code file, you must save the code file in the 'cartographer' directory.
                     - 'list_working_directory_files': to list all files in the working directory.

                     The user's instruction are: {map_task}

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": mapState["messages"]
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(create_llm("CODE_WRITER_LLM"), [list_working_directory_files, execute_code_file, execute_code], state_schema=MapState, state_modifier=_modify_mapState)
        response = agent.invoke(mapState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="CodeDebugger")

        mapState["messages"].append(ai_message)
        mapState["is_task_finish"] = "CONTINUE"
        return mapState

    def finish_gis_code(mapState: MapState) -> MapState:
        '''
        结束代码生成
        '''
        logger = get_logger_with_session()
        logger.info("Processing agent: cartographer_agent(CodeFinisher)")
        def _modify_mapState(mapState: MapState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                     You are *CodeFinisher*, the concluding member of a collaborative multi-agent system designed for generating Python code for GIS cartography. Other agents in this system include [*DataReader*, *CodeWriter*, *CodeReviewer*, *CodeDebugger*], each playing a crucial role in the pipeline.
                     Your primary responsibility is to provide a comprehensive summary of the system's outcomes, focusing on the final Python code file and the generated images.

                     Your Step-by-Step Process:
                     1. Summarize the Final Code File:
                        - Provide an overview of the final Python code file, including:
                            - Its purpose and intended functionality.
                            - Key operations performed in the code, especially GIS-related tasks (e.g., spatial joins, buffers, projections).
                            - Improvements or revisions made during the debugging and review stages.
                     2. Use 'save_file_metadata' tool to save the final code file metadata and the generated data results. You're part of the 'cartographer' role.
                     3. Summarize the Generated Images:
                        - You have to be truthful. You can't make things up. If the image is not successfully generated, skip this step.
                        - List and describe all images generated during the system run, including:
                            - The type of image (e.g., png, jpg, jpeg).
                            - Key statistics or summaries of the data (e.g., number of records, attributes, or spatial properties).
                            - The file paths to the output images.
                        - Highlight the relevance of the outputs to the overall objective of the system.
                     4. You must use 'create_image_review_queue_file' tool to create a txt file to add all images to the review queue. If no images are generated, pass in an empty list to 'create_image_review_queue_file'.
                     
                     Output Requirements:
                     1. A section summarizing the final Python code file.
                     2. A section summarizing all generated images with file paths.
                     3. A final evaluation of the system’s performance, noting any areas of improvement.
                     4. Which images did you add to the review queue in the txt file.

                     """),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": mapState["messages"]
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, create_image_review_queue_file, save_file_metadata], state_schema=MapState, state_modifier=_modify_mapState)
        response = agent.invoke(mapState)
        ai_message = AIMessage(content=_extract_message_content(response), name="CodeFinisher")

        mapState["messages"].append(ai_message)
        mapState["is_task_finish"] = "FINISH"
        return mapState
    
    workflow = StateGraph(MapState)
    workflow.add_node("read_gis_metadata", read_gis_metadata)
    workflow.add_node("write_code", write_gis_code)
    workflow.add_node("review_code", review_gis_code)
    workflow.add_node("debug_code", debug_gis_code)
    workflow.add_node("finish_code", finish_gis_code)

    workflow.add_edge(START, "read_gis_metadata")
    workflow.add_edge("read_gis_metadata", "write_code")
    workflow.add_edge("write_code", "review_code")
    workflow.add_edge("review_code", "debug_code")
    workflow.add_edge("debug_code", "finish_code")

    return workflow.compile()

def create_reporter_agent(
    llm: ChatOpenAI, 
) -> CompiledStateGraph:
    logger = get_logger_with_session()
    

    def reporter(reportState: ReportState):
        logger.info("Processing agent: reporter_agent")

        report_dir = os.path.join(os.environ["WORKING_DIRECTORY"], "reporter")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_task = reportState["report_task"]
        logger.info(f"report_task: {report_task}")

        def _modify_reportState(reportState: ReportState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                    """
                    You are *Reporter*, an experienced scientific writer tasked with drafting comprehensive reports.

                    Your primary duties include:
                    1. Clearly stating the research hypothesis and objectives in the introduction.
                    2. Detailing the methodology used, including data collection and analysis techniques.
                    3. Structuring the report into coherent sections (e.g., Introduction, Methodology, Results, Discussion, Conclusion).
                    4. Synthesizing information from various sources into a unified narrative.
                    5. Integrating relevant data visualizations and ensuring they are appropriately referenced and explained.
                    6. If *Cartographer* has generated image results, you need to insert images into the document at the appropriate location using the syntax "![description](xxx.png)", and add a description to the image.


                    Constraints:
                    - Focus solely on report writing; do not perform data analysis or create visualizations.
                    - Maintain an objective, academic tone throughout the report.
                    - Cite all sources using APA style and ensure that all findings are supported by evidence.

                    **Your Step-by-Step Process:**
                    1. Use 'query_files_by_role' tool to query all files generated by the *Coder* and *Cartographer* role, confirm the file path.
                    2. Fully understand *Planner*'s program, *Coder*'s data analysis results, and *Cartographer*'s image results.
                    3. Use 'write_document' tool to create a new report file named, The report files you write must be saved in 'reporter' folder.
                    4. Use 'save_file_metadata' tool to save the final report file metadata and the generated data results.

                    The user's instruction is: {report_task}

                    使用中文回答。使用中文撰写文档。
                    """
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": reportState.get("messages", []),  # 从state获取消息历史
                "report_task": report_task,
            }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [write_document, query_files_by_role, save_file_metadata], state_schema=ReportState, state_modifier=_modify_reportState)
        response = agent.invoke(reportState)
        ai_message = AIMessage(content=_extract_message_content(response), name="reporter")

        reportState["messages"].append(ai_message)
        reportState["is_task_finish"] = "FINISH"
        return reportState
    
    workflow = StateGraph(ReportState)
    workflow.add_node("reporter", reporter)
    workflow.add_edge(START, "reporter")
    workflow.add_edge("reporter", END)
    return workflow.compile()

def create_codeReviewer_agent(
    llm: ChatOpenAI, 
) -> CompiledStateGraph:
    def codeReviewer(codeReviewState: CodeReviewState):
        logger = get_logger_with_session()
        logger.info("Processing agent: codeReviewer")

        code_task = codeReviewState["code_task"]
        logger.info(f"current_code_task: {code_task}")

        def _modify_codeReviewState(codeReviewState: CodeReviewState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                    """
                    You are *Reviewer*, a specialized AI assistant in a GIS SPATIAL ANALYSIS TEAM. The system also includes other agents like [*Manager*, *Coder*]. Your role is to verify that *Coder* has produced the expected result file as instructed by *Manager* and reply CONTINUE or REVISION, making the system go to the next step. You will receive a Json message from the Manager to the Coder and feedback from the Coder to the Manager.

                    *Your Step-by-Step Process*:
                    1. Using the 'list_working_directory_files' tool to check if the working directory contains the expected data analysis results, as specified in the *Manager*'s instructions. Or use the 'find_file_in_working_directory' tool to check if the working directory contains the expected data analysis results.
                    2. Sending your review results to the Manager. If the required file (e.g., Shapefile files like `polygon.shp`) is found, respond with 'CONTINUE' as a prefix. If the file is missing or incomplete, you must ask *Coder* to re-execute the task to produce the missing results, respond with 'REVISION' as a prefix.

                    *Your Response Example*: 
                    Example1: CONTINUE: The required file `roughness_index.tif` was found in ...
                    Example2: REVISION: The required file `roughness_index.tif` is missing...
                    
                    *Guidelines*:
                    1. You are only responsible for verifying *Coder*'s output files based on the given instructions. Do not attempt to review any other parts of the project.
                    2. You have no authority to decide if the overall task is complete; you can only determine if *Coder* has correctly completed the assigned subtask.
                    3. *Coder* is not responsible for tasks like image generation, visualization, or mapping. You cannot ask *Coder* to generate images.
                    4.To perform your review, immediately use the 'list_working_directory_files' tool to check the working directory. You do not need to describe the process step-by-step, simply verify the required files are present, then take action accordingly.

                    Don't put off your inspection efforts, you must implement the tools now.
                    """
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                    ("user", 
                    """
                    The task assigned by *Manager* is: {code_task}. 
                    Now, Calling tool: list_working_directory_files, and only after you give a complete review will you stop work.
                    """
                    ),
                ]
            )
            state_variables = {
                "messages": codeReviewState.get("messages", []),
                "code_task": codeReviewState.get("code_task", ""),
            }
            response = prompt.invoke(state_variables).to_messages()
            return response

        agent = create_react_agent(create_llm(agent_name="CODE_REVIEW_LLM"), [list_working_directory_files, find_file_in_working_directory], state_schema=CodeReviewState, state_modifier=_modify_codeReviewState)
        response = agent.invoke(codeReviewState)
        content = _extract_message_content(response)
        logger.info(content)
        ai_message = AIMessage(content=content, name="codeReviewer")
        codeReviewState["messages"].append(ai_message)
        codeReviewState["is_task_finish"] = "FINISH"
        return codeReviewState
    
    workflow = StateGraph(CodeReviewState)
    workflow.add_node("codeReviewer", codeReviewer)
    workflow.add_edge(START, "codeReviewer")
    workflow.add_edge("codeReviewer", END)
    return workflow.compile()


def create_mapReviewer_agent(
    llm: ChatOpenAI,
    tools: list[tool], 
    system_prompt: str,
    state: State,
    working_directory: str = None
    ) -> CompiledStateGraph:
    
    # 先思考要审查的图片和要求，之后调用模型审查。
    class ImageReviewRequest(TypedDict):
        """Current image and its requirements."""
        task: Annotated[str | None, ..., "User and Manager's task"]
        task_requirement: Annotated[str | None, ..., "User and Manager's task requirements"]
        visualization_requirement: Annotated[str | None, ..., "User and Manager's visualization requirements"]
    
    def review_image_prepare(imageReviewState: ImageReviewState) -> ImageReviewState:
        logger = get_logger_with_session()

        # 获取当前要审查的图片路径
        current_image_path = get_next_image_from_queue()
        logger.info(f"当前要准备要求的图片路径：{current_image_path}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    You are an expert in visualization task review in the GIS field. You want to review the results of cartographer's images, usually located in the 'cartographer' folder.

                    Your current image path to review is: {current_image_path}
                        
                    Your Step by Step Process:
                    1. Write down the Manager's task for this image.
                    2. Write down the User and Manager's requirements for this image.
                     - We divide the requirements into two types: task requirements and visualization requirements. 
                     - Task requirements refer to whether the information reflected in the image meets the requirements.
                     - Visualization requirements refer to whether the visual effects of the image meet the requirements, which can refer to the following format and example.
                    Format:
                    - task: *Manager*'s task.
                    - task_requirement: User and *Manager*'s requirements for the task results.
                    - visualization_requirement: User and *Manager*'s requirements for the image elements such as color, legend, compass, text, etc. (if any).
                    
                    Example:
                    - task: 1. Load the output Shapefiles 'min_area_rectangle.shp' and 'min_convex_hull.shp'. 2. Create visualizations to clearly depict the minimum area rectangles and minimum convex hulls overlaid on the original point groups. 3. Ensure that the visualizations effectively communicate the key findings.
                    - task_requirement: The image should reflect the shape of the point set being enclosed by the minimum convex hull, that is, the distribution of the point set and the shape of the minimum convex hull should be displayed.
                    - visualization_requirement: Include axes, legends, no compass, and the colors need to be bright.
                    
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # 准备状态变量
        state_variables = {
            "messages": imageReviewState.get("messages", [])  # 从state获取消息历史
        }
        structured_llm = llm.with_structured_output(ImageReviewRequest)
        # 调用模型
        response = structured_llm.invoke(prompt.invoke(state_variables)) # 返回ImageReviewRequest对象
        task = response["task"]
        task_requirement = response["task_requirement"]
        visualization_requirement = response["visualization_requirement"]
        logger.info(f"task: {task}")
        logger.info(f"task_requirement: {task_requirement}")
        logger.info(f"visualization_requirement: {visualization_requirement}")
        logger.info(f"当前要审查的图片路径：{current_image_path}, 要求：{response}")
        ai_message = AIMessage(content=f"当前要审查的图片路径：{current_image_path}, 要求：{response} ", name="MapReviewer")
        
        # 更新imageState
        imageReviewState["messages"].append(ai_message)
        imageReviewState["current_image_path"] = current_image_path
        imageReviewState["task"] = task
        imageReviewState["task_requirement"] = task_requirement
        imageReviewState["visualization_requirement"] = visualization_requirement
        
        return imageReviewState

    def review_image(imageReviewState: ImageReviewState) -> ImageReviewState:
        logger = get_logger_with_session()

        # 获取图片路径
        image_path = imageReviewState.get("current_image_path", "")
        task_requirement = imageReviewState.get("task_requirement", "")
        visualization_requirement = imageReviewState.get("visualization_requirement", "")

        logger.info("--------------- 当前审核图片信息 ---------------")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Task requirements: {task_requirement}")
        logger.info(f"Visualization requirements: {visualization_requirement}")

        # 图片编码
        working_directory = os.environ["WORKING_DIRECTORY"]
        if working_directory in image_path:
            full_image_path = image_path
        else:
            full_image_path = os.path.join(working_directory, image_path)
        with open(full_image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    f"""
                    You are *MapReviewer*, an expert in visualization task review in the GIS field.
                    Your tasks include:
                    1. You need to check if *Cartographer* has correctly completed the task assigned by *Manager*. The *process_decision* is the current *Manager*'s instruction.
                    2. You need to check if the working directory contains the visualization results expected by *Manager*. If the visualization results are missing (such as image files), ask *Cartographer* to re-execute the task.
                    3. You need to check if the images meet the requirements specified by *Manager*.
                    4. You cannot read pdf files, nor do you need to read pdf files; you can only read image files in formats such as ['png', 'jpeg', 'gif', 'webp'].
                    5. You have no authority to determine if the overall task is complete; you can only judge whether *Cartographer* has correctly completed the assigned subtask.

                    After your review, if revisions are needed, respond with 'REVISION' as a prefix, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix.
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"现在审核上传的图片：{image_path}，这是要求：{task_requirement},  {visualization_requirement}"},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                )
            ]
        )
        state_variables = {
            "messages": imageReviewState["messages"]  # 从state获取消息历史
        }       
        # 调用模型
        response = llm.invoke(prompt.invoke(state_variables))
        content = _extract_message_content(response)
        ai_message = AIMessage(content=content, name="MapReviewer")
        logger.info(f"map_reviewer_agent 图片审查结果：{content}")

        # 更新状态
        imageReviewState["messages"].append(ai_message)
        imageReviewState["image_review_results"] = {
            **imageReviewState.get("image_review_results", {}),
            imageReviewState.get("current_image_path", ""): content
        }
    
        return imageReviewState
    
    def review_image_finish(imageReviewState: ImageReviewState) -> ImageReviewState:
        logger = get_logger_with_session()

        prompt = ChatPromptTemplate.from_messages(
            [
                # (
                #     "system", 
                #     f"""
                #     You are *MapReviewer*, an expert in visualization task review in the GIS field.
                #     Your tasks include:
                #     1. You need to check if *Cartographer* has correctly completed the task assigned by *Manager*. The *process_decision* is the current *Manager*'s instruction.
                #     2. You need to check if the working directory contains the visualization results expected by *Manager*. If the visualization results are missing (such as image files), ask *Cartographer* to re-execute the task.
                #     3. You need to check if the images meet the requirements specified by *Manager*.
                #     4. You cannot read pdf files, nor do you need to read pdf files; you can only read image files in formats such as ['png', 'jpeg', 'gif', 'webp'].
                #     5. You have no authority to determine if the overall task is complete; you can only judge whether *Cartographer* has correctly completed the assigned subtask.

                #     After your review, if revisions are needed, respond with 'REVISION' as a prefix, set needs_revision=True, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix and set needs_revision=False.
                #     """
                # ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    """
                    Given the conversation above, Summarize the results of your review for each image in detail, don't omit them. if revisions are needed, respond with 'REVISION' as a prefix, set needs_revision=True, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix.

                    If Cartographer is unable to generate an image because it lacks sufficient data, it has also responded with the 'CONTINUE' prefix.
                    """
                ),
            ]
        )
        # 准备状态变量
        state_variables = {
            "messages": imageReviewState.get("messages", [])  # 从state获取消息历史
        }
        response = llm.invoke(prompt.invoke(state_variables)) # 返回正常的字典
        content = _extract_message_content(response)
        ai_message = AIMessage(content=content, name="MapReviewer")
        logger.info(f"图片审查汇总结果：{content}")
        # 裁剪消息列表，保留列表最初两条消息和最后一条消息
        messages = imageReviewState["messages"]
        imageReviewState["messages"] = messages[:2] + [ai_message]
        return imageReviewState
    
    def review_router(imageReviewState: ImageReviewState) -> str:
        logger = get_logger_with_session()
        image_queue_count = get_image_queue_count()
        logger.info(f"图片审查队列剩余数量：{image_queue_count}")
        if isinstance(image_queue_count, int) and image_queue_count > 0:
            return "CONTINUE"
        else:
            return "FINISH"

    workflow = StateGraph(State)

    workflow.add_node("review_image_prepare", review_image_prepare)

    workflow.add_node("review_image", review_image)
    workflow.add_node("review_image_finish", review_image_finish)

    workflow.add_conditional_edges(
        START,
        review_router,
        {
            "CONTINUE": "review_image_prepare",
            "FINISH": "review_image_finish",
        }
    )
    workflow.add_edge("review_image_prepare", "review_image")
    workflow.add_edge("review_image", "review_image_finish")
    workflow.add_conditional_edges(
        "review_image_finish",
        review_router,
        {
            "CONTINUE": "review_image_prepare",
            "FINISH": END,
        }
    )

    return workflow.compile()

def create_reportReviewer_agent(
    llm: ChatOpenAI, 
) -> CompiledStateGraph:
    logger = get_logger_with_session()
    
    def reportReviewer(reportReviewerState: ReportReviewState):
        logger.info("Processing agent: reportReviewer_agent")
        current_report_task = reportReviewerState.get("report_task", "")
        logger.info(f"current_report_task: {current_report_task}")

        reporter_files_info = query_files_by_role.invoke("Reporter")
        logger.info(f"reporter_files_info: {reporter_files_info}")

        def _modify_reportReviewerState(reportReviewerState: ReportReviewState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                    """
                    You are *ReportReviewer*, an expert in writing task review in the GIS field.
                    
                    Your tasks include:
                    1. You need to check if *Reporter* has correctly completed the task assigned by *Manager*. The *process_decision* is the current *Manager*'s instruction.
                    2. You need to check if the working directory contains the report results expected by *Manager*. If the report results are missing, ask *Reporter* to re-execute the task.
                    3. You have no authority to determine if the overall task is complete; you can only judge whether *Reporter* has correctly completed the assigned subtask.

                    After your review, if revisions are needed, respond with 'REVISION' as a prefix, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix.

                    The task assigned by *Manager* is: {current_report_task}

                    The report file created by *Reporter* is: {reporter_files_info}

                    """
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            state_variables = {
                "messages": reportReviewerState["messages"],
                "current_report_task": current_report_task,
                "reporter_files_info": reporter_files_info
                }
            response = prompt.invoke(state_variables).to_messages()
            return response
        
        agent = create_react_agent(llm, [list_working_directory_files, read_document, save_file_metadata], state_schema=ReportReviewState, state_modifier=_modify_reportReviewerState)
        response = agent.invoke(reportReviewerState)
        ai_message = AIMessage(content=_extract_message_content(response), name="ReportReviewer")
        reportReviewerState["messages"].append(ai_message)
        reportReviewerState["is_last_step"] = True
        return reportReviewerState
    
    workflow = StateGraph(ReportReviewState)
    workflow.add_node("reportReviewer", reportReviewer)
    workflow.add_edge(START, "reportReviewer")
    workflow.add_edge("reportReviewer", END)
    return workflow.compile()

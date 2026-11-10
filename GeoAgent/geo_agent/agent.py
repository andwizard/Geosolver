import os
from .state import CodeReviewState, ReportReviewState, ReportState, State
from .create_agent import create_cartographer_agent, create_codeReviewer_agent, create_manager_agent, create_coder_agent, create_mapReviewer_agent, create_planner_agent, create_reportReviewer_agent, create_reporter_agent

from model.llm import create_llm
from geo_agent_tools.code_tools import execute_code, execute_command, python_repl, query_function_document
from geo_agent_tools.doc_tools import create_document, read_document, write_document, edit_document
from geo_agent_tools.gis_tools import reading_gis_metadata, read_shapefile_info
from geo_agent_tools.io_tools import create_directory, find_file_in_working_directory, query_files_by_role

WORKING_DIRECTORY = os.environ["WORKING_DIRECTORY"]

planner = create_planner_agent(
    create_llm("PLAN_LLM"), 
    [],
    '',
    State,
    WORKING_DIRECTORY
)

manager = create_manager_agent(
    create_llm("MANAGER_LLM"),
    '',
    ["Coder", "Cartographer", "Reporter"],
    [read_document]
)


# # planner消融用
# manager_wo_planner = create_manager_agent_wo_planner(
#     create_llm("MANAGER_LLM"),
#     '',
#     ["Coder", "Cartographer", "Reporter"],
#     []
# )

coder = create_coder_agent(
    create_llm("CODE_LLM"),
    [],
    '',
    State,
    WORKING_DIRECTORY
)

cartographer = create_cartographer_agent(
    create_llm("VISUALIZATION_LLM"),
    [], 
    '',
    State,
    WORKING_DIRECTORY
    )

reporter = create_reporter_agent(
    create_llm("REPORT_LLM"), 
)

codeReviewer = create_codeReviewer_agent(
    create_llm("CODE_REVIEW_LLM"), 
)

mapReviewer = create_mapReviewer_agent(
    create_llm("VISUALIZATION_LLM"), 
    [],
    '',
    State,
    WORKING_DIRECTORY
    )

reportReviewer = create_reportReviewer_agent(
    create_llm("REPORT_REVIEW_LLM"), 
)

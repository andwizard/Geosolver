from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langgraph.managed import IsLastStep
from langgraph.graph.state import CompiledStateGraph
from openai import InternalServerError
from .create_agent import create_planner_agent
from geo_agent.state import CodeReviewState, ImageReviewState, ImageReviewState, ManageState, MapState, PlanState, ReportReviewState, ReportState, State, CodeState
from geo_agent.agent import planner, manager, coder, reporter, cartographer, codeReviewer, mapReviewer, reportReviewer
from utils.logger import get_logger_with_session
from utils.utils import _extract_message_content
import json
import re
import os
from pathlib import Path

def _extract_current_task(last_message: BaseMessage) -> str:
    """
    Extract the task from the last message.
    """
    last_message_content = last_message.content
    if last_message.name == "manager":
        if isinstance(last_message_content, str):
            try:
                task_dict = json.loads(last_message_content)
                task = task_dict.get("task", last_message_content)
            except json.JSONDecodeError:
                try:
                    task_dict = eval(last_message_content)
                    task = task_dict.get("task", last_message_content)
                except:
                    task = ""

    else:
        try:
            task_dict = json.loads(last_message_content)
            task = task_dict.get("task", last_message_content)
        except json.JSONDecodeError:
            try:
                task_dict = eval(last_message_content)
                task = task_dict.get("task", last_message_content)
            except:
                task = "" 
    return task

def _extract_subtask_num(last_message: BaseMessage) -> str:
    """
    Extract the subtask number from the last message.
    """
    subtask_num = ""
    last_message_content = last_message.content
    if last_message.name == "manager":
        if isinstance(last_message_content, str):
            try:
                task_dict = json.loads(last_message_content)
                subtask_num = str(task_dict.get("subtask_num", ""))
            except json.JSONDecodeError:
                try:
                    task_dict = eval(last_message_content)
                    subtask_num = str(task_dict.get("subtask_num", ""))
                except:
                    subtask_num = ""
    return subtask_num

def planner_node(state: State) -> State:
    """
    Process an planner agent's action and update the state accordingly.
    """

    agent = planner
    name = "planner"

    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        planState = PlanState(
            messages = state.get("messages", []),
            is_last_step = IsLastStep()
        )
        logger.debug(f"PlanState: {planState}")

        result = agent.invoke(planState)
        logger.debug(f"Agent {name} result: {result}")
        ai_message = AIMessage(content=_extract_message_content(result), name=name)

        state["messages"].append(ai_message)
        state["sender"] = name

        state["task_category"] = result["task_category"]
        state["allowed_roles"] = result["allowed_roles"]

        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}    

def manager_node(state: State) -> State:
    """
    Process an manager agent's action and update the state accordingly.
    """
    
    agent = manager
    name = "manager"

    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        manageState = ManageState(
            messages = state.get("messages", []),
            is_last_step = IsLastStep()
        )
        logger.info(f"ManageState: {manageState}")

        result = agent.invoke(manageState)
        logger.info(f"Agent {name} result: {result}")
        process_decision = _extract_message_content(result)
        ai_message = AIMessage(content=process_decision, name=name)

        state["messages"].append(ai_message)
        state["sender"] = name
        state["process_decision"] = process_decision
        state["current_subtask_num"] = _extract_subtask_num(ai_message)
        state["current_code_task"] = _extract_current_task(ai_message)
        state["current_map_task"] = _extract_current_task(ai_message)
        state["current_report_task"] = _extract_current_task(ai_message)
        
        logger.info(f"current_subtask_num: {state['current_subtask_num']}")
        logger.info(f"current_code_task: {state['current_code_task']}")
        logger.info(f"current_map_task: {state['current_map_task']}")
        logger.info(f"current_report_task: {state['current_report_task']}")
        logger.info(f"Agent {name} processing completed")
        logger.info(f"State: {state}")
        return state
    except Exception as e:
        logger.info(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def coder_node(state: State) -> State:
    """
    Process an code agent's action and update the state accordingly.
    """

    agent = coder 
    name = "coder"

    
    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        messages = state.get("messages", [])
        
        last_message = messages[-1]
        code_task = _extract_current_task(last_message)
        subtask_num = _extract_subtask_num(last_message)
        logger.info(f"subtask_num: {subtask_num}")
        codeState = CodeState(
            messages = HumanMessage(content=f"{code_task}"),
            subtask_num = subtask_num,
            reviewer_comment = "",
            code_task = code_task,
            is_last_step = IsLastStep()
        )
        logger.debug(f"CodeState: {codeState}")

        result = agent.invoke(codeState)
        logger.debug(f"Agent {name} result: {result}")
        ai_message = AIMessage(content=_extract_message_content(result), name=name)

        state["messages"].append(ai_message)
        state["sender"] = name
        state["code_state"] = _extract_message_content(result)
        state["current_code_task"] = code_task

        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def cartographer_node(state: State) -> State:
    """
    Process an cartographer agent's action and update the state accordingly.
    """

    agent = cartographer
    name = "cartographer"
    
    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        messages = state.get("messages", [])
        
        last_message = messages[-1]
        map_task = _extract_current_task(last_message)

        mapState = MapState(
            messages = HumanMessage(content=f"{map_task}"),
            map_task = map_task,
            is_last_step = IsLastStep()
        )
        logger.debug(f"MapState: {mapState}")

        result = agent.invoke(mapState)
        logger.debug(f"Agent {name} result: {result}")
        ai_message = AIMessage(content=_extract_message_content(result), name=name)

        state["messages"].append(ai_message)
        state["sender"] = name
        state["cartographer_state"] = _extract_message_content(result)
        state["current_map_task"] = map_task

        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def reporter_node(state: State) -> State:
    
    
    agent = reporter
    name = "reporter"
    

    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        messages = state.get("messages", [])

        last_message = messages[-1]
        report_task = _extract_current_task(last_message)

        reportState = ReportState(
            messages = messages,
            report_task = report_task,
            is_last_step = IsLastStep(),
        )
        logger.debug(f"ReportState: {reportState}")

        result = agent.invoke(reportState)
        logger.debug(f"Agent {name} result: {result}")
        ai_message = AIMessage(content=_extract_message_content(result), name=name)
        
        state["messages"].append(ai_message)
        state["sender"] = name
        state["report_state"] = _extract_message_content(result)
        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def codeReviewer_node(state: State) -> State:
    
        
    agent = codeReviewer
    name = "codeReviewer"
    
    
    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        code_task = state.get("current_code_task", "")
        messages = state.get("messages", [])
        codeReviewState = CodeReviewState(
            messages = messages[-1:],
            code_task = code_task,
            is_last_step = IsLastStep(),
        )
        logger.debug(f"CodeReviewState: {codeReviewState}")
        result = agent.invoke(codeReviewState)
        logger.debug(f"Agent {name} result: {result}")

        last_message = messages[-1] if messages else None
        if 'REVISION' in str(last_message.content):
            content = (
                f"{_extract_message_content(result)}\n"
                f"The task assigned by *Manager* is:{code_task}\n"
                f"You are now going to re-execute the task as required above."
            )
        else:
            content = _extract_message_content(result)
        ai_message = AIMessage(content=content, name=name)
        
        state["messages"].append(ai_message)
        state["sender"] = name
        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def mapReviewer_node(state: State) -> State:
    
    agent = mapReviewer
    name = "mapReviewer"

    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        current_map_task = state.get("current_map_task", "")
        messages = state.get("messages", [])
        imageReviewState = ImageReviewState(
            messages = messages[-2:],
            current_image_path = "",
            task = current_map_task,
            task_requirement = "",
            visualization_requirement = "",
            image_review_results = {},
            is_last_step = IsLastStep(),
        )
        logger.debug(f"ImageReviewState: {imageReviewState}")

        result = agent.invoke(imageReviewState)
        logger.debug(f"Agent {name} result: {result}")
        content = (
            f"{_extract_message_content(result)}\n"
            f"The task assigned by *Manager* is:{current_map_task}\n"
            f"You are now going to re-execute the task as required above."
        )
        ai_message = AIMessage(content=content, name=name)
        
        state["messages"].append(ai_message)
        state["sender"] = name
        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

def reportReviewer_node(state: State) -> State:
    
    agent = reportReviewer
    name = "reportReviewer"
    
    logger = get_logger_with_session()
    logger.info(f"Processing agent: {name}")
    try:
        messages = state.get("messages", [])
        last_message = messages[-1]
        report_task = _extract_current_task(last_message)
        
        reportReviewState = ReportReviewState(
            messages = messages[-2:],
            report_task = report_task,
            is_last_step = IsLastStep(),
        )
        logger.debug(f"ReportReviewState: {reportReviewState}")

        result = agent.invoke(reportReviewState)
        logger.debug(f"Agent {name} result: {result}")
        ai_message = AIMessage(content=_extract_message_content(result), name=name)
        
        state["messages"].append(ai_message)
        state["sender"] = name
        logger.info(f"Agent {name} processing completed")
        return state
    except Exception as e:
        logger.error(f"Error occurred while processing agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": [error_message]}

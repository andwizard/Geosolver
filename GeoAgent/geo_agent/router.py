from .state import State
from typing import Literal
from langchain_core.messages import AIMessage
from utils.logger import get_logger_with_session
import json
import re
import ast

NodeType = Literal["Planner", "Manager", "Coder", "Reporter", "Cartographer", "CodeReviewer", "MapReviewer", "Recorder",  "Refiner"]
ProcessNodeType = Literal['Coder', 'Reporter', 'Manager', 'Cartographer']

def process_router(state: State) -> ProcessNodeType:
    logger = get_logger_with_session()
    logger.info("Entering process_router")
    
    process_decision = state.get("process_decision", "")
    
    logger.debug(f"process_decision: {process_decision}")
    logger.debug(f"process_decision_type: {type(process_decision)}")

    if len(state["messages"]) >= 20:
        logger.info("Messages length >= 20. Ending process.")
        return "END"
    
    if isinstance(process_decision, AIMessage):
        logger.debug("Process decision is an AIMessage")
        try:

            content = re.sub(r"(?<!\w)'(?!\w)", '"', process_decision.content)
            decision_dict = json.loads(content)
            process_decision = decision_dict.get('next', '')
            logger.debug(f"Parsed process decision from JSON: {process_decision}")
        except json.JSONDecodeError:
            process_decision = process_decision.content
            logger.warning("Failed to parse process decision as JSON. Using content directly.")
    elif isinstance(process_decision, dict):
        process_decision = process_decision.get('next', '') 
        logger.debug(f"Process decision is a dictionary. Using 'next' value: {process_decision}")
    elif not isinstance(process_decision, str):
        process_decision = str(process_decision)
        logger.warning(f"Unexpected process decision type. Converting to string: {process_decision}")
    
    if isinstance(process_decision, str):
        try:
            decision_dict = json.loads(process_decision.replace("'", '"'))
            process_decision = decision_dict.get('next', '')
            logger.debug(f"Parsed process decision from JSON: {process_decision}")
        except json.JSONDecodeError:
            decision_dict = ast.literal_eval(process_decision)
            process_decision = decision_dict.get('next', '')
            logger.debug(f"Parsed process decision from dictionary: {process_decision}")

    valid_decisions = {"Coder", "Reporter", "Cartographer"}
    
    if process_decision in valid_decisions:
        logger.info(f"Valid process decision: {process_decision}")
        return process_decision
    
    if process_decision == "FINISH":
        logger.info("Process decision is FINISH. Ending process.")

        return "END"
    
    if not process_decision or process_decision not in valid_decisions:
        logger.warning(f"Invalid or empty process decision: {process_decision}. Defaulting to 'Manager'.")
        return "Manager"
    
    logger.info("Defaulting to 'Manager'")
    return "Manager"

def quality_review_router(state: State) -> NodeType:
    logger = get_logger_with_session()
    logger.info("Entering quality review router")
    
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if (last_message and 'REVISION' in str(last_message.content)) or state.get("needs_revision", False):
        current_node = state.get("sender", "")
        revision_routes = {
            "codeReviewer": "Coder", 
            "mapReviewer": "Cartographer",
            "reportReviewer": "Reporter", 
        }
        result = revision_routes.get(current_node, "Recorder")
        logger.info(f"Revision needed. Routing to: {result}")
        return result
    
    else:
        return "Recorder"

from typing_extensions import TypedDict, Annotated, Sequence, Union
import operator
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # User
    user_query: str = ""
    user_guide: str = ""

    # Planner
    task_category: str = ""
    task_plan: str = ""
    allowed_roles: list[str] = []

    # Manager
    process_decision: str = ""
    current_subtask_num: int = 0
    current_code_task: str = ""
    current_map_task: str = ""
    current_report_task: str = ""
    
    # Worker
    coder_state: str = ""
    reporter_state: str = ""
    cartographer_state: str = ""

    # Reviewer
    code_reviewer_state: str = ""
    map_reviewer_state: str = ""
    report_reviewer_state: str = ""
    
    needs_revision: bool = False

    # The identifier of the agent who sent the last message
    sender: str = ""

    is_last_step: IsLastStep

class PlanState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task_category: str = ""
    task_plan: str = ""
    plan_file_path: str = ""
    allowed_roles: list[str] = []
    is_last_step: IsLastStep

class ManageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep

class CodeState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] 
    messages_history: list[AnyMessage] = []
    code_task: str = ""
    reviewer_comment: str = ""
    subtask_num: int = 0
    is_task_finish: str = ""
    is_last_step: IsLastStep

class MapState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    messages_history: list[AnyMessage] = []
    map_task: str = ""
    subtask_num: int = 0
    is_task_finish: str = ""
    is_last_step: IsLastStep

class ReportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    report_task: str = ""
    subtask_num: int = 0
    is_last_step: IsLastStep

class CodeReviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    code_task: str = ""
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep

class ImageReviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_image_path: str = ""
    task: str = ""
    task_requirement: str = ""
    visualization_requirement: str = ""
    image_review_results: dict[str, str] = {}
    is_last_image: str = ""
    is_last_step: IsLastStep

class ReportReviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    report_task: str = ""
    is_last_step: IsLastStep
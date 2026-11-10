from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from geo_agent.state import State
from geo_agent.node import planner_node, manager_node, coder_node, cartographer_node, reporter_node, codeReviewer_node, mapReviewer_node, reportReviewer_node
from geo_agent.router import process_router, quality_review_router

# Original Graph
def create_workflow():
    # Create state graph for the workflow
    workflow = StateGraph(State)

    # Plan Node
    workflow.add_node("Planner", planner_node)

    # Process Node
    workflow.add_node("Manager", manager_node)
    workflow.add_node("Coder", coder_node)
    workflow.add_node("Cartographer", cartographer_node)
    workflow.add_node("Reporter", reporter_node)

    # Review Node
    workflow.add_node("CodeReviewer", codeReviewer_node)
    workflow.add_node("MapReviewer", mapReviewer_node)
    workflow.add_node("ReportReviewer", reportReviewer_node)

    workflow.add_edge("Planner", "Manager")

    workflow.add_conditional_edges(
        "Manager",
        process_router,
        {
            "Coder": "Coder",
            "Reporter": "Reporter",
            "Cartographer": "Cartographer",
            "Manager": "Manager",
            "END": END,
        }
    )

    workflow.add_edge("Coder", "CodeReviewer")
    workflow.add_edge("Cartographer", "MapReviewer")
    workflow.add_edge("Reporter", "ReportReviewer")

    for member in ["CodeReviewer", "MapReviewer", "ReportReviewer"]:
        workflow.add_conditional_edges(
            member,
            quality_review_router,
            {
                "Coder": "Coder",
                "Cartographer": "Cartographer",
                "Reporter": "Reporter",

                "Recorder": "Manager",
            }
        )
    workflow.set_entry_point("Planner")

    return workflow

def create_graph(thread_id: str = "1"):
    graph_config = {"configurable": {"thread_id": thread_id}}
    memory = MemorySaver()
    workflow = create_workflow()
    graph = workflow.compile(checkpointer=memory)
    return graph, graph_config

def create_workflow_wo_reviewer():
    # Create state graph for the workflow
    workflow = StateGraph(State)

    # Plan Node
    workflow.add_node("Planner", planner_node)

    # Process Node
    workflow.add_node("Manager", manager_node)
    workflow.add_node("Coder", coder_node)
    workflow.add_node("Cartographer", cartographer_node)
    workflow.add_node("Reporter", reporter_node)

    workflow.add_edge("Planner", "Manager")

    workflow.add_conditional_edges(
        "Manager",
        process_router,
        {
            "Coder": "Coder",
            "Reporter": "Reporter",
            "Cartographer": "Cartographer",
            "Manager": "Manager",
            "END": END,
        }
    )

    for member in ["Coder", "Reporter", "Cartographer"]:
        workflow.add_edge(member, "Manager")
    workflow.set_entry_point("Planner")
    
    return workflow

def create_workflow_wo_planner():
    # Create state graph for the workflow
    workflow = StateGraph(State)

    # Process Node
    workflow.add_node("Manager", manager_wo_planner_node)
    workflow.add_node("Coder", coder_node)
    workflow.add_node("Cartographer", cartographer_node)
    workflow.add_node("Reporter", reporter_node)

    # Review Node
    workflow.add_node("CodeReviewer", codeReviewer_node)
    workflow.add_node("MapReviewer", mapReviewer_node)
    workflow.add_node("ReportReviewer", reportReviewer_node)

    workflow.add_conditional_edges(
        "Manager",
        process_router,
        {
            "Coder": "Coder",
            "Reporter": "Reporter",
            "Cartographer": "Cartographer",
            "Manager": "Manager",

            "END": END,
        }
    )

    workflow.add_edge("Coder", "CodeReviewer")
    workflow.add_edge("Cartographer", "MapReviewer")
    workflow.add_edge("Reporter", "ReportReviewer")

    for member in ["CodeReviewer", "MapReviewer", "ReportReviewer"]:
        workflow.add_conditional_edges(
            member,
            quality_review_router,
            {
                "Coder": "Coder",
                "Cartographer": "Cartographer",
                "Reporter": "Reporter",
                "Recorder": "Manager",
            }
        )

    workflow.set_entry_point("Manager")

    return workflow

def create_graph_wo_reviewer(thread_id: str = "1"):
    graph_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    memory = MemorySaver()
    workflow = create_workflow_wo_reviewer()
    graph = workflow.compile(checkpointer=memory)
    return graph, graph_config

def create_graph_wo_planner(thread_id: str = "1"):
    graph_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    memory = MemorySaver()
    workflow = create_workflow_wo_planner()
    graph = workflow.compile(checkpointer=memory)
    return graph, graph_config
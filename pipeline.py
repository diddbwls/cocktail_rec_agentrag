from langgraph.graph import StateGraph, START, END
from nodes.task_classifier import query_classification
from nodes.retriever import graph_query_node, incremental_retriever
from nodes.reflection import reflection
from nodes.generator import generator
from utils.types import PipelineState
from typing import Dict, Any, Literal
from nodes.user_question import initial_query


def build_pipeline_graph() -> StateGraph:
    """
    LangGraph pipeline configuration
    
    flow:
    START → user_question → task_classification → retriever (Round 1) → reflection → [conditional branching]
                                                        ↓
    generator ← [>=80 or 3 rounds] ← reflection ← incremental_retriever (Round 2,3) ← [<80 & <3 rounds]
        ↓
       END
    """
    graph = StateGraph(PipelineState)
    
    # node definition
    graph.add_node("task_classification", query_classification)
    graph.add_node("user_question", initial_query)
    graph.add_node("retriever", graph_query_node)  # initial search (Round 1)
    graph.add_node("incremental_retriever", incremental_retriever)  # incremental search (Round 2, 3)
    graph.add_node("reflection", reflection)
    graph.add_node("generator", generator)
    
    # basic flow
    graph.add_edge(START, "user_question")  # Updated edge name
    graph.add_edge("user_question", "task_classification")  # Updated edge name
    graph.add_edge("task_classification", "retriever")  # Round 1: initial search
    graph.add_edge("retriever", "reflection")
    graph.add_edge("incremental_retriever", "reflection")  # Round 2, 3: incremental search after reflection
    
    # conditional branching based on reflection result
    def reflection_condition(state: Dict[str, Any]) -> Literal["score<80", "score>=80"]:
        """
        Conditional branching based on reflection result
        
        Args:
            state: pipeline state
            
        Returns:
            "score<80": incremental_retriever for retry (Round 2, 3)
            "score>=80": generator for completion
        """
        # Handle missing 'should_retry' field
        if "should_retry" not in state:
            state["should_retry"] = False
            
        should_retry = state.get("should_retry", False)
        score = state.get("score", 0)
        iteration_count = state.get("iteration_count", 0)
        
        # log output
        print(f"🔀 conditional branching decision:")
        print(f"   - score: {score:.1f}")
        print(f"   - iteration count: {iteration_count}/3")
        print(f"   - retry needed: {should_retry}")
        
        if should_retry:
            print("   → retry: incremental_retriever for retry")
            return "score<80"
        else:
            print("   → completion: generator for completion")
            return "score>=80"
    
    # add conditional edges
    graph.add_conditional_edges(
        "reflection",
        reflection_condition,
        {
            "score<80": "incremental_retriever",    # incremental search for retry
            "score>=80": "generator"                   # completion
        }
    )
    
    # final output
    graph.add_edge("generator", END)
    
    return graph

def save_workflow_diagram():
    """save workflow diagram as PNG"""
    try:
        import os
        from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
        
        # create workflow
        app = build_pipeline_graph().compile()
        
        # graph_viz directory creation (absolute path)
        viz_dir = "./langgraph/graph_viz"
        os.makedirs(viz_dir, exist_ok=True)
        
        # create and save diagram
        diagram_path = os.path.join(viz_dir, "workflow.png")
        
        print("🔄 creating workflow diagram...")
        
        # get Mermaid code and modify
        mermaid_code = app.get_graph().draw_mermaid()
        
        # change reflection -> generator edge from dashed to solid (keep label)
        mermaid_code = mermaid_code.replace(
            "reflection -. &nbsp;score>=80&nbsp; .-> generator;",
            "reflection --&nbsp;score>=80&nbsp;--> generator;"
        )
        
        # method 1: pyppeteer local rendering (try first)
        try:
            print("  🌐 method 1: local browser rendering")
            # create PNG with modified mermaid code
            from langchain_core.runnables.graph_mermaid import draw_mermaid_png
            graph_image = draw_mermaid_png(
                mermaid_code,
                draw_method=MermaidDrawMethod.PYPPETEER,
                max_retries=3,
                retry_delay=1.0
            )
            with open(diagram_path, 'wb') as f:
                f.write(graph_image)
            print(f"✅ workflow diagram updated (local rendering): {diagram_path}")
            return
            
        except Exception as local_error:
            print(f"  ⚠️ local rendering failed: {local_error}")
            
            # method 2: API rendering (fallback)
            try:
                print("  📡 method 2: mermaid.ink API")
                graph_image = app.get_graph().draw_mermaid_png(
                    max_retries=5,
                    retry_delay=2.0
                )
                with open(diagram_path, 'wb') as f:
                    f.write(graph_image)
                print(f"✅ workflow diagram updated (API): {diagram_path}")
                return
                
            except Exception as api_error:
                print(f"  ❌ API method also failed: {api_error}")
                raise Exception(f"Local: {local_error}\nAPI: {api_error}")
        
    except Exception as e:
        print(f"❌ diagram creation failed: {e}")
        
def run_pipeline(user_query: str, image_path: str = None) -> Dict[str, Any]:
    """
    pipeline execution helper function
    
    Args:
        user_query: user query
        image_path: image path (optional)
        
    Returns:
        state dictionary containing execution result
    """
    # create pipeline graph
    graph = build_pipeline_graph()
    app = graph.compile()
    
    # create initial state
    initial_state = {
        "user_query": {
            "text": user_query,
            "image": image_path
        },
        "input_text": user_query,
        "input_image": image_path,
        "current_top_k": 3,
        "iteration_count": 0,
        "debug_info": {}
    }
    
    print(f"🚀 pipeline started")
    print(f"📝 question: {user_query}")
    if image_path:
        print(f"🖼️ image: {image_path}")
    
    try:
        # execute pipeline
        result = app.invoke(initial_state)
        
        print(f"✅ pipeline completed")
        return result
        
    except Exception as e:
        print(f"❌ pipeline execution error: {e}")
        return {
            "final_text": f"system error occurred: {str(e)}",
            "final_cocktails": [],
            "error": str(e)
        }

if __name__ == "__main__":
    
    #  execution options
    
    # Option 1: pipeline test run
    # test_pipeline()
    
    # Option 2: workflow diagram PNG 
    save_workflow_diagram()
    

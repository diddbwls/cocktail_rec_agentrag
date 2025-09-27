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
    LangGraph 파이프라인 구성
    
    플로우:
    START → user_question → task_classification → retriever (Round 1) → reflection → [조건부 분기]
                                                        ↓
    generator ← [80점 이상 or 3회 반복] ← reflection ← incremental_retriever (Round 2,3) ← [80점 미만 & 3회 미만]
        ↓
       END
    """
    graph = StateGraph(PipelineState)
    
    # 노드 정의
    graph.add_node("task_classification", query_classification)
    graph.add_node("user_question", initial_query)
    graph.add_node("retriever", graph_query_node)  # 초기 검색 (Round 1)
    graph.add_node("incremental_retriever", incremental_retriever)  # 증분 검색 (Round 2, 3)
    graph.add_node("reflection", reflection)
    graph.add_node("generator", generator)
    
    # 기본 플로우
    graph.add_edge(START, "user_question")  # Updated edge name
    graph.add_edge("user_question", "task_classification")  # Updated edge name
    graph.add_edge("task_classification", "retriever")  # Round 1: 초기 검색
    graph.add_edge("retriever", "reflection")
    graph.add_edge("incremental_retriever", "reflection")  # Round 2, 3: 증분 검색 후 reflection
    
    # Reflection에서 조건부 분기
    def reflection_condition(state: Dict[str, Any]) -> Literal["score<80", "score>=80"]:
        """
        Reflection 결과에 따른 조건부 라우팅
        
        Args:
            state: 파이프라인 상태
            
        Returns:
            "score<80": incremental_retriever로 재시도 (Round 2, 3)
            "score>=80": generator로 종료
        """
        # Handle missing 'should_retry' field
        if "should_retry" not in state:
            state["should_retry"] = False
            
        should_retry = state.get("should_retry", False)
        score = state.get("score", 0)
        iteration_count = state.get("iteration_count", 0)
        
        # 로그 출력
        print(f"🔀 조건부 분기 판단:")
        print(f"   - 점수: {score:.1f}")
        print(f"   - 반복 횟수: {iteration_count}/3")
        print(f"   - 재시도 필요: {should_retry}")
        
        if should_retry:
            print("   → 재시도: incremental_retriever로 이동")
            return "score<80"
        else:
            print("   → 완료: generator로 이동")
            return "score>=80"
    
    # 조건부 엣지 추가
    graph.add_conditional_edges(
        "reflection",
        reflection_condition,
        {
            "score<80": "incremental_retriever",    # 증분 검색으로 재시도
            "score>=80": "generator"                   # 완료
        }
    )
    
    # 최종 출력
    graph.add_edge("generator", END)
    
    return graph

def save_workflow_diagram():
    """워크플로우 다이어그램을 PNG로 저장"""
    try:
        import os
        from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
        
        # 워크플로우 생성
        app = build_pipeline_graph().compile()
        
        # graph_viz 디렉토리 생성 (절대 경로 사용)
        viz_dir = "./langgraph/graph_viz"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 다이어그램 생성 및 저장
        diagram_path = os.path.join(viz_dir, "workflow.png")
        
        print("🔄 워크플로우 다이어그램 생성 중...")
        
        # Mermaid 코드 가져와서 수정
        mermaid_code = app.get_graph().draw_mermaid()
        
        # reflection -> generator 엣지를 점선에서 실선으로 변경 (라벨 유지)
        mermaid_code = mermaid_code.replace(
            "reflection -. &nbsp;score>=80&nbsp; .-> generator;",
            "reflection --&nbsp;score>=80&nbsp;--> generator;"
        )
        
        # 방법 1: pyppeteer 로컬 렌더링 (우선 시도)
        try:
            print("  🌐 방법 1: 로컬 브라우저 렌더링")
            # 수정된 mermaid 코드로 PNG 생성
            from langchain_core.runnables.graph_mermaid import draw_mermaid_png
            graph_image = draw_mermaid_png(
                mermaid_code,
                draw_method=MermaidDrawMethod.PYPPETEER,
                max_retries=3,
                retry_delay=1.0
            )
            with open(diagram_path, 'wb') as f:
                f.write(graph_image)
            print(f"✅ 워크플로우 다이어그램이 업데이트되었습니다 (로컬 렌더링): {diagram_path}")
            return
            
        except Exception as local_error:
            print(f"  ⚠️ 로컬 렌더링 실패: {local_error}")
            
            # 방법 2: API 렌더링 (fallback)
            try:
                print("  📡 방법 2: mermaid.ink API")
                graph_image = app.get_graph().draw_mermaid_png(
                    max_retries=5,
                    retry_delay=2.0
                )
                with open(diagram_path, 'wb') as f:
                    f.write(graph_image)
                print(f"✅ 워크플로우 다이어그램이 업데이트되었습니다 (API): {diagram_path}")
                return
                
            except Exception as api_error:
                print(f"  ❌ API 방법도 실패: {api_error}")
                raise Exception(f"Local: {local_error}\nAPI: {api_error}")
        
    except Exception as e:
        print(f"❌ 다이어그램 생성 실패: {e}")
        
def run_pipeline(user_query: str, image_path: str = None) -> Dict[str, Any]:
    """
    파이프라인 실행 헬퍼 함수
    
    Args:
        user_query: 사용자 질문
        image_path: 이미지 경로 (선택사항)
        
    Returns:
        실행 결과가 포함된 상태 딕셔너리
    """
    # 파이프라인 그래프 생성
    graph = build_pipeline_graph()
    app = graph.compile()
    
    # 초기 상태 생성
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
    
    print(f"🚀 파이프라인 시작")
    print(f"📝 질문: {user_query}")
    if image_path:
        print(f"🖼️ 이미지: {image_path}")
    
    try:
        # 파이프라인 실행
        result = app.invoke(initial_state)
        
        print(f"✅ 파이프라인 완료")
        return result
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 오류: {e}")
        return {
            "final_text": f"시스템 오류가 발생했습니다: {str(e)}",
            "final_cocktails": [],
            "error": str(e)
        }

if __name__ == "__main__":
    
    #  실행 옵션
    
    # Option 1: 파이프라인 테스트 실행
    # test_pipeline()
    
    # Option 2: 워크플로우 다이어그램 PNG 생성
    save_workflow_diagram()
    
    # Option 3: 커스텀 쿼리로 파이프라인 테스트
    # custom_query = "your query here"
    # result = run_pipeline(custom_query)
    # print(f"\n최종 결과: {result.get('final_text', '응답 없음')}")
    

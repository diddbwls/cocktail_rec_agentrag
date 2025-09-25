from typing import Dict, Any, List
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Retrieval 클래스들 임포트
from retrieval.c1_retrieval import C1Retrieval
from retrieval.c2_retrieval import C2Retrieval
from retrieval.c3_retrieval import C3Retrieval
from retrieval.c4_retrieval import C4Retrieval

# Retrieval 시스템 인스턴스 (전역으로 관리하여 재사용)
retrieval_systems = {
    "C1": None,
    "C2": None,
    "C3": None,
    "C4": None
}

def get_retrieval_system(task_type: str):
    """
    태스크 타입에 따른 검색 시스템 인스턴스 반환
    싱글톤 패턴으로 관리하여 매번 재생성하지 않음
    """
    global retrieval_systems
    
    if retrieval_systems[task_type] is None:
        print(f"🔄 {task_type} 검색 시스템 초기화 중...")
        if task_type == "C1":
            retrieval_systems[task_type] = C1Retrieval(use_python_config=True)
        elif task_type == "C2":
            retrieval_systems[task_type] = C2Retrieval(use_python_config=True)
        elif task_type == "C3":
            retrieval_systems[task_type] = C3Retrieval(use_python_config=True)
        elif task_type == "C4":
            retrieval_systems[task_type] = C4Retrieval(use_python_config=True)
        print(f"✅ {task_type} 검색 시스템 초기화 완료")
    
    return retrieval_systems[task_type]

def graph_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    태스크 타입에 따라 적절한 검색 알고리즘을 실행하는 통합 검색 노드
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        검색 결과가 포함된 상태 딕셔너리
    """
    # 태스크 타입 가져오기
    task_type = state.get("task_type", "C1")
    query_text = state.get("input_text_with_image", state.get("input_text", ""))
    current_top_k = state.get("current_top_k", 3)
    
    print(f"\n🔍 검색 시작: {task_type} (Top-{current_top_k})")
    print(f"📝 쿼리: {query_text}")
    
    try:
        # 해당 태스크의 검색 시스템 가져오기
        retrieval_system = get_retrieval_system(task_type)
        
        # top-k 설정 업데이트 (Reflection에서 증가시킬 수 있음)
        if hasattr(retrieval_system, 'config'):
            retrieval_system.config['final_top_k'] = current_top_k
            print(f"🔧 Top-K 업데이트: {current_top_k}")
        
        # 검색 실행
        results = retrieval_system.retrieve(query_text)
        
        # 결과 저장 (초기 검색이므로 cumulative_results도 초기화)
        state["search_results"] = results
        state["cumulative_results"] = results.copy()  # 누적 결과 초기화
        
        print(f"✅ 초기 검색 완료: {len(results)}개 칵테일 발견")
        
        # 디버그 정보 추가
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["retrieval"] = {
            "task_type": task_type,
            "query_length": len(query_text),
            "results_count": len(results),
            "top_k": current_top_k,
            "cocktail_names": [r.get("name", "Unknown") for r in results[:5]]  # 상위 5개만
        }
        
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        state["search_results"] = []
        
        # 오류 정보 저장
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["retrieval_error"] = str(e)
    
    return state

def incremental_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    증분 검색을 실행하는 노드 (Round 2, 3용)
    기존 결과에 추가로 필요한 만큼만 검색
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        증분 검색 결과가 포함된 상태 딕셔너리
    """
    # 필요한 정보 가져오기
    task_type = state.get("task_type", "C1")
    query_text = state.get("input_text_with_image", state.get("input_text", ""))
    current_top_k = state.get("current_top_k", 3)
    cumulative_results = state.get("cumulative_results", [])
    
    print(f"\n🔍 증분 검색 시작: {task_type} (현재: {len(cumulative_results)}개 → 목표: {current_top_k}개)")
    
    # 이미 충분한 결과가 있는지 확인
    if len(cumulative_results) >= current_top_k:
        print(f"✅ 이미 충분한 결과 보유: {len(cumulative_results)}개")
        state["search_results"] = cumulative_results[:current_top_k]
        return state
    
    try:
        # 추가로 필요한 개수 계산
        additional_needed = current_top_k - len(cumulative_results)
        print(f"🔧 추가 검색 필요: {additional_needed}개")
        
        # 해당 태스크의 검색 시스템 가져오기
        retrieval_system = get_retrieval_system(task_type)
        
        # 더 많은 결과를 검색하여 추가 개수 확보
        # 여유분을 두어 current_top_k + 2개 검색
        extended_top_k = current_top_k + 2
        if hasattr(retrieval_system, 'config'):
            retrieval_system.config['final_top_k'] = extended_top_k
            print(f"🔧 확장 Top-K 설정: {extended_top_k}")
        
        # 전체 검색 실행 (기존 로직 재사용)
        extended_results = retrieval_system.retrieve(query_text)
        
        # 기존 결과와 겹치지 않는 새로운 결과만 추출
        existing_names = {cocktail.get('name', '') for cocktail in cumulative_results}
        new_results = [cocktail for cocktail in extended_results 
                      if cocktail.get('name', '') not in existing_names]
        
        # 필요한 만큼만 추가
        additional_results = new_results[:additional_needed]
        
        # 누적 결과 업데이트
        updated_cumulative = cumulative_results + additional_results
        state["cumulative_results"] = updated_cumulative
        state["search_results"] = updated_cumulative[:current_top_k]
        
        print(f"✅ 증분 검색 완료: +{len(additional_results)}개 추가 (총 {len(updated_cumulative)}개)")
        print(f"📋 현재 라운드 결과: {len(state['search_results'])}개")
        
        # 디버그 정보 업데이트
        if "debug_info" not in state:
            state["debug_info"] = {}
        
        state["debug_info"]["incremental_retrieval"] = {
            "task_type": task_type,
            "target_top_k": current_top_k,
            "previous_count": len(cumulative_results),
            "additional_needed": additional_needed,
            "additional_found": len(additional_results),
            "total_cumulative": len(updated_cumulative),
            "current_round_count": len(state['search_results'])
        }
        
    except Exception as e:
        print(f"❌ 증분 검색 오류: {e}")
        # 오류 시 기존 결과 유지
        state["search_results"] = cumulative_results[:current_top_k]
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["incremental_retrieval_error"] = str(e)
    
    return state

# 개별 검색 함수들 (필요 시 직접 호출용)
def c1_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    C1: 색상-ingredient 시각적 검색 알고리즘
    """
    state["task_type"] = "C1"
    return graph_query_node(state)

def c2_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    C2: glasstype-ingredient 매칭
    """
    state["task_type"] = "C2"
    return graph_query_node(state)

def c3_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    C3: multi-hop ingredient 확장 검색
    """
    state["task_type"] = "C3"
    return graph_query_node(state)

def c4_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    C4: 재료기반 유사 레시피 칵테일 추천
    """
    state["task_type"] = "C4"
    return graph_query_node(state)
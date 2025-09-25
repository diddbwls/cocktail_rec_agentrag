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
    
    try:
        # 해당 태스크의 검색 시스템 가져오기
        retrieval_system = get_retrieval_system(task_type)
        
        # top-k 설정 업데이트 (Reflection에서 증가시킬 수 있음)
        if hasattr(retrieval_system, 'config'):
            retrieval_system.config['final_top_k'] = current_top_k
            print(f"🔧 Top-K 업데이트: {current_top_k}")
        
        # 검색 실행
        retrieval_result = retrieval_system.retrieve(query_text)
        
        # C1, C3, C4는 dict를 반환, C2는 리스트를 반환
        if isinstance(retrieval_result, dict):
            # dict 형태: {'results': [...], 'full_ranked_names': [...], 'current_top_k': N}
            results = retrieval_result['results']
            state["full_ranked_cocktails"] = retrieval_result['full_ranked_names']
            print(f"✅ 전체 랭킹 저장: {len(state['full_ranked_cocktails'])}개 (캐싱용)")
            # 디버깅: 저장된 내용 확인
            print(f"📋 저장된 칵테일 이름들: {state['full_ranked_cocktails']}")
        else:
            # C2의 경우 기존 방식 (리스트 반환)
            results = retrieval_result
            state["full_ranked_cocktails"] = []  # C2는 캐싱 미지원
            print(f"✅ C2 방식: 캐싱 미지원")
        
        # 결과 저장
        state["search_results"] = results
        
        print(f"✅ 초기 검색 완료: {len(results)}개 칵테일 발견")
        
        # 디버그 정보 추가
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["retrieval"] = {
            "task_type": task_type,
            "query_length": len(query_text),
            "results_count": len(results),
            "top_k": current_top_k,
            "has_full_ranking": len(state["full_ranked_cocktails"]) > 0,
            "full_ranking_count": len(state["full_ranked_cocktails"]),
            "cocktail_names": [r.get("name", "Unknown") for r in results[:5]]  # 상위 5개만
        }
        
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        state["search_results"] = []
        state["full_ranked_cocktails"] = []
        
        # 오류 정보 저장
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["retrieval_error"] = str(e)
    
    return state

def incremental_retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    증분 검색을 실행하는 노드 (Round 2, 3용)
    캐시된 랭킹에서 필요한 만큼만 선택 (C1, C3, C4) 또는 기존 방식 (C2)
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        증분 검색 결과가 포함된 상태 딕셔너리
    """
    # 필요한 정보 가져오기
    task_type = state.get("task_type", "C1")
    current_top_k = state.get("current_top_k", 3)
    full_ranked_names = state.get("full_ranked_cocktails", [])
    
    print(f"\n🔍 증분 검색 시작: {task_type} (목표: Top-{current_top_k})")
    
    # 디버깅 정보 출력
    print(f"🔍 캐시 상태 확인: task_type={task_type}, cache_size={len(full_ranked_names)}")
    if full_ranked_names:
        print(f"📋 캐시된 칵테일들: {full_ranked_names}") 
    
    # C2의 경우 기존 방식 사용 (캐싱 미지원)
    if task_type == "C2":
        print(f"📋 C2 태스크: 캐싱 없이 기존 방식 사용")
        return _original_incremental_logic(state)
    
    # C1, C3, C4인데 캐시가 없는 경우
    if not full_ranked_names:
        print(f"⚠️ {task_type} 태스크이지만 캐시가 비어있음, 기존 방식으로 폴백")
        # state에 있는 다른 키들도 확인
        print(f"🔍 State 키 확인: {list(state.keys())}")
        # full_ranked_cocktails 키가 있는지 명시적으로 확인
        if "full_ranked_cocktails" in state:
            print(f"⚠️ full_ranked_cocktails 키는 존재하지만 비어있음")
        else:
            print(f"⚠️ full_ranked_cocktails 키가 state에 없음!")
        return _original_incremental_logic(state)
    
    # C1, C3, C4의 경우 캐시된 랭킹 사용
    try:
        if len(full_ranked_names) >= current_top_k:
            # 캐시된 전체 랭킹에서 필요한 만큼만 선택
            selected_names = full_ranked_names[:current_top_k]
            retrieval_system = get_retrieval_system(task_type)
            results = retrieval_system.get_cocktail_details(selected_names)
            state["search_results"] = results
            
            print(f"✅ 캐시된 랭킹에서 선택: {len(results)}개 (Top-{current_top_k})")
            print(f"📋 전체 캐시 크기: {len(full_ranked_names)}개")
            
        else:
            print(f"⚠️ 랭킹 부족 ({len(full_ranked_names)} < {current_top_k}), 기존 결과 유지")
            # 가능한 만큼만 선택
            if full_ranked_names:
                retrieval_system = get_retrieval_system(task_type)
                results = retrieval_system.get_cocktail_details(full_ranked_names)
                state["search_results"] = results
            
        # 디버그 정보 업데이트
        if "debug_info" not in state:
            state["debug_info"] = {}
        
        state["debug_info"]["incremental_retrieval"] = {
            "task_type": task_type,
            "method": "cached_ranking",
            "target_top_k": current_top_k,
            "cache_size": len(full_ranked_names),
            "selected_count": len(state.get("search_results", []))
        }
        
    except Exception as e:
        print(f"❌ 캐시 기반 증분 검색 오류: {e}")
        # 폴백: 기존 방식
        return _original_incremental_logic(state)
    
    return state


def _original_incremental_logic(state: Dict[str, Any]) -> Dict[str, Any]:
    """캐시를 사용할 수 없는 경우를 위한 기존 증분 검색 로직"""
    # 필요한 정보 가져오기
    task_type = state.get("task_type", "C1")
    query_text = state.get("input_text_with_image", state.get("input_text", ""))
    current_top_k = state.get("current_top_k", 3)
    existing_results = state.get("search_results", [])
    
    print(f"🔄 {task_type} 기존 방식 증분 검색: (현재: {len(existing_results)}개 → 목표: {current_top_k}개)")
    
    try:
        # 해당 태스크의 검색 시스템 가져오기
        retrieval_system = get_retrieval_system(task_type)
        
        # top-k 증가시켜서 새로 검색
        if hasattr(retrieval_system, 'config'):
            retrieval_system.config['final_top_k'] = current_top_k
            print(f"🔧 Top-K 업데이트: {current_top_k}")
        
        # 전체 검색 재실행
        results = retrieval_system.retrieve(query_text)
        
        # 리스트 형태의 결과 처리
        if isinstance(results, dict):
            results = results['results']
        
        state["search_results"] = results
        print(f"✅ {task_type} 증분 검색 완료: {len(results)}개")
        
        # 디버그 정보 업데이트
        if "debug_info" not in state:
            state["debug_info"] = {}
        
        state["debug_info"]["incremental_retrieval"] = {
            "task_type": task_type,
            "method": "full_research",
            "target_top_k": current_top_k,
            "results_count": len(results)
        }
        
    except Exception as e:
        print(f"❌ {task_type} 기존 방식 증분 검색 오류: {e}")
        # 오류 시 기존 결과 유지
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
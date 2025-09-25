from typing import Any, Dict, Optional, List, TypedDict

class PipelineState(TypedDict, total=False):
    """
    LangGraph 파이프라인 전체 상태를 dict 형태로 정의
    total=False로 설정하여 모든 필드가 선택적
    """
    # -------------------
    # 사용자 입력
    # -------------------
    user_query: Dict[str, Any]            # 예: {"text": "질문 내용", "image": <이미지 경로>}
    input_text: str                       # 텍스트만 별도로 저장
    input_image: Optional[str]            # 이미지 경로
    
    # -------------------
    # Task Classification
    # -------------------
    task_type: str                        # C1~C4 태스크 분류 결과
    task_confidence: float                # 분류 신뢰도
    task_reason: str                      # 분류 이유
    
    # -------------------
    # Retriever 관련
    # -------------------
    embedding_model: str                  # 임베딩 모델명
    input_text_with_image: str            # 이미지 설명이 포함된 전체 텍스트
    current_top_k: int                    # 현재 검색 top-k 값
    search_results: List[Dict[str, Any]]  # 현재 라운드 검색 결과 리스트
    cumulative_results: List[Dict[str, Any]]  # 누적 검색 결과 (증분 검색용)
    full_ranked_cocktails: List[str]      # 전체 유사도 랭킹된 칵테일 이름 리스트 (캐싱용)
    
    # -------------------
    # Reflection 관련
    # -------------------
    iteration_count: int                  # 현재 반복 횟수 (최대 3)
    score: float                          # 전체 평가 점수
    evaluation_scores: Dict[str, float]   # 세부 평가 점수
    # {
    #     "relevance": 0.0-100.0,
    #     "diversity": 0.0-100.0,
    #     "completeness": 0.0-100.0,
    #     "coherence": 0.0-100.0
    # }
    best_result: Dict[str, Any]           # 최고 점수 결과 저장
    reflection_feedback: str              # 리플렉션 피드백 메시지
    
    # -------------------
    # Generator 결과
    # -------------------
    final_text: str                       # 최종 응답 텍스트
    final_cocktails: List[Dict[str, Any]]  # 최종 추천 칵테일 리스트
    
    # -------------------
    # 비교 시스템 (초기 vs 최종)
    # -------------------
    initial_search_results: List[Dict[str, Any]]  # Round 1 검색 결과 (3개)
    initial_response: str                         # Round 1 생성 답변
    final_search_results: List[Dict[str, Any]]    # 최고 점수 라운드 검색 결과
    final_response: str                           # 최고 점수 라운드 생성 답변
    
    # 최고 점수 라운드 정보
    final_best_score: float                       # 최고 점수
    final_best_round: int                         # 최고 점수를 받은 라운드 번호
    final_best_top_k: int                         # 최고 점수를 받은 때의 Top-K
    
    # -------------------
    # 디버깅/로깅
    # -------------------
    debug_info: Dict[str, Any]            # 디버깅 정보        
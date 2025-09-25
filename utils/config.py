"""
칵테일 AI 검색 시스템 설정
"""
from prompts.c1_prompt import C1_KEYWORD_EXTRACTION_PROMPT, C1_SYSTEM_MESSAGE
from prompts.c2_prompt import C2_KEYWORD_EXTRACTION_PROMPT, C2_SYSTEM_MESSAGE
from prompts.c3_prompt import C3_KEYWORD_EXTRACTION_PROMPT, C3_SYSTEM_MESSAGE
from prompts.c4_prompt import C4_KEYWORD_EXTRACTION_PROMPT, C4_SYSTEM_MESSAGE

# 기본 모델 설정
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE_FILE = "embedding_cache.json"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

#top -k 설정
INITIAL_TOP_K = 6
FINAL_TOP_K = 3

# C1 태스크 설정 (색상-재료 기반 알고리즘)
C1_CONFIG = {
    "initial_top_k": INITIAL_TOP_K,        # 초기 imageDescription 유사도 검색 결과 수
    "final_top_k": FINAL_TOP_K,          # 최종 결과 수
    "similarity_threshold": 0.7,
    "keyword_extraction_prompt": C1_KEYWORD_EXTRACTION_PROMPT,
    "system_message": C1_SYSTEM_MESSAGE
}

# C2 태스크 설정 (Glass Type + 재료 매칭)
C2_CONFIG = {
    "initial_top_k": INITIAL_TOP_K,                # 초기 name_embedding 유사도 검색 결과 수
    "final_top_k": FINAL_TOP_K,                  # 최종 결과 수
    "min_candidates_threshold": 2,     # 재료 줄이기 시작하는 최소 후보 수
    "target_candidates": 5,            # 목표 후보 칵테일 수
    "keyword_extraction_prompt": C2_KEYWORD_EXTRACTION_PROMPT,
    "system_message": C2_SYSTEM_MESSAGE
}

# C3 태스크 설정 (Multi-hop 재료 확장)
C3_CONFIG = {
    "initial_top_k": INITIAL_TOP_K,                  # 초기 재료 기반 검색 결과 수
    "final_top_k": FINAL_TOP_K,                    # 최종 결과 수
    "expansion_top_k": 8,                # Multi-hop 확장 검색 결과 수
    "min_ingredient_match": 1,           # 최소 재료 매치 수
    "min_cocktail_usage": 2,             # 재료가 최소 사용되어야 하는 칵테일 수
    "name_similarity_threshold": 0.7,    # 이름 유사도 임계값
    "keyword_extraction_prompt": C3_KEYWORD_EXTRACTION_PROMPT,
    "system_message": C3_SYSTEM_MESSAGE
}

# C4 태스크 설정 (재료기반 유사 레시피 칵테일 추천)
C4_CONFIG = {
    "initial_top_k": INITIAL_TOP_K,                  # 관계 기반 검색 결과 수 (여유분)
    "final_top_k": FINAL_TOP_K,                    # 최종 결과 수
    "complexity_tolerance": 2,           # 재료 개수 차이 허용 범위
    "min_shared_ingredients": 1,         # 최소 공유 재료 수
    "name_similarity_threshold": 0.7,    # 이름 유사도 임계값 (폴백용)
    "embedding_fallback_top_k": 3,       # 임베딩 폴백 검색 수
    "keyword_extraction_prompt": C4_KEYWORD_EXTRACTION_PROMPT,
    "system_message": C4_SYSTEM_MESSAGE
}

# 전체 설정을 딕셔너리로 내보내기 (하위 호환성)
CONFIG = {
    "embedding_model": EMBEDDING_MODEL,
    "embedding_cache_file": EMBEDDING_CACHE_FILE,
    "model": LLM_MODEL,
    "temperature": TEMPERATURE,
    "c1_config": C1_CONFIG,
    "c2_config": C2_CONFIG,
    "c3_config": C3_CONFIG,
    "c4_config": C4_CONFIG
}

def get_config():
    """설정 딕셔너리 반환"""
    return CONFIG

def get_c1_config():
    """C1 설정만 반환"""
    return C1_CONFIG

def get_c2_config():
    """C2 설정만 반환"""
    return C2_CONFIG

def get_c3_config():
    """C3 설정만 반환"""
    return C3_CONFIG

def get_c4_config():
    """C4 설정만 반환"""
    return C4_CONFIG

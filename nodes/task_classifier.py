from utils.types import PipelineState
from utils.openai_client import OpenAIClient
from utils.prompt_loader import PromptLoader
import json
import os
from typing import Dict, Any
import sys
from nodes.user_question import initial_query

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from prompts.task_classifier_prompt import TASK_CLASSIFIER_TEMPLATE

# OpenAI 클라이언트 초기화
openai_client = OpenAIClient()

# PromptLoader 초기화 (폴백용)
prompt_loader = PromptLoader()


def query_classification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 쿼리를 C1-C4 태스크 중 하나로 분류
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        태스크 분류 결과가 포함된 상태 딕셔너리
    """
    # 먼저 initial_query 실행하여 이미지 처리
    state = initial_query(state)
    
    # 분류할 텍스트 가져오기
    query_text = state.get("input_text_with_image", state.get("input_text", ""))
    
    if not query_text:
        print("❌ 분류할 쿼리가 없습니다")
        state["task_type"] = "C1"  # 기본값
        state["task_confidence"] = 0.0
        state["task_reason"] = "쿼리가 비어있어 기본값으로 설정"
        return state
    
    try:
        # 태스크 분류 프롬프트 로드 (직접 임포트 사용)
        try:
            classifier_prompt = TASK_CLASSIFIER_TEMPLATE
        except:
            # 폴백: PromptLoader 사용
            classifier_prompt = prompt_loader.get_classifier_prompt()
        
        # 프롬프트에 쿼리 삽입
        prompt = classifier_prompt.replace("{question}", query_text)
        
        print(f"🤖 태스크 분류 중...")
        print(f"📝 쿼리: {query_text}")
        
        # OpenAI API 호출 (JSON 응답 요청)
        response = openai_client.generate(prompt, response_format="json")
        
        # JSON 파싱
        result = openai_client.parse_json_response(response)
        
        # 결과 검증 및 저장
        task_type = result.get("task", "C1")
        confidence = float(result.get("confidence", 0))
        reason = result.get("reason", "")
        
        # 유효성 검증
        if task_type not in ["C1", "C2", "C3", "C4"]:
            print(f"⚠️ 잘못된 태스크 타입: {task_type}, C1으로 대체")
            task_type = "C1"
            
        # 상태 업데이트
        state["task_type"] = task_type
        state["task_confidence"] = confidence
        state["task_reason"] = reason
        
        # 초기 top-k 설정
        state["current_top_k"] = 3  # 기본값
        state["iteration_count"] = 0
        
        print(f"✅ 태스크 분류 완료: {task_type} (신뢰도: {confidence}%)")
        print(f"📋 분류 이유: {reason}")
        
        # 디버그 정보 추가
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["classification"] = {
            "task": task_type,
            "confidence": confidence,
            "reason": reason,
            "raw_response": response if "error" in result else None
        }
        
    except Exception as e:
        print(f"❌ 태스크 분류 오류: {e}")
        # 오류 시 기본값 설정
        state["task_type"] = "C1"
        state["task_confidence"] = 0.0
        state["task_reason"] = f"분류 오류로 기본값 사용: {str(e)}"
        state["current_top_k"] = 3
        state["iteration_count"] = 0
    
    return state
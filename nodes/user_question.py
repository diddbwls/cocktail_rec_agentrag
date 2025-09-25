import os
import sys
import base64
from typing import Dict, Any, Optional

from utils.openai_client import OpenAIClient
from utils.prompt_loader import PromptLoader
from prompts.query_image_prompt import QUERY_IMAGE_PROMPT


# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# OpenAI 클라이언트 초기화
openai_client = OpenAIClient()

# PromptLoader 초기화 (폴백용)
prompt_loader = PromptLoader()

def describe_image(image_path: str, prompt: Optional[str] = None) -> str:
    """
    이미지를 분석하고 설명을 생성하는 함수
    
    Args:
        image_path: 분석할 이미지 파일 경로
        prompt: 사용할 프롬프트 (기본값: QUERY_IMAGE_PROMPT)
        
    Returns:
        이미지에 대한 설명 텍스트
    """
    try:
        # 전역 OpenAI 클라이언트 사용
        client = openai_client.client
        
        # 프롬프트 설정
        if prompt is None:
            prompt = QUERY_IMAGE_PROMPT
        
        # 이미지 파일 존재 확인
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        # 이미지를 base64로 인코딩
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # 파일 확장자로 MIME 타입 결정
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext in ['.png']:
            mime_type = "image/png"
        elif ext in ['.webp']:
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # 기본값
        
        # GPT-4o-mini 호출
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        return resp.choices[0].message.content
        
    except Exception as e:
        print(f"이미지 설명 생성 오류: {e}")
        return f"이미지를 분석할 수 없습니다: {str(e)}"


def initial_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 입력을 처리하고 이미지가 있을 경우 설명을 생성하여 텍스트와 결합
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    # user_query에서 텍스트와 이미지 추출
    user_query = state.get("user_query", {})
    text = user_query.get("text", "")
    image_path = user_query.get("image", None)
    
    # 기본값 설정
    state["input_text"] = text
    state["input_image"] = image_path
    state["input_text_with_image"] = text
    
    # 이미지가 있으면 설명 생성
    if image_path and os.path.exists(image_path):
        try:
            print(f"🖼️ 이미지 설명 생성 중: {image_path}")
            image_description = describe_image(image_path)
            
            # 이미지 설명과 텍스트 결합
            if image_description:
                combined_text = f"{image_description} {text}" if text else image_description
                state["input_text_with_image"] = combined_text
                print(f"✅ 이미지 설명 생성 완료")
            else:
                print("⚠️ 이미지 설명 생성 실패")
                
        except Exception as e:
            print(f"❌ 이미지 처리 오류: {e}")
            state["input_text_with_image"] = text
    
    # 디버그 정보 추가
    if "debug_info" not in state:
        state["debug_info"] = {}
    state["debug_info"]["initial_query"] = {
        "has_image": bool(image_path),
        "text_length": len(text),
        "combined_text_length": len(state["input_text_with_image"])
    }
    
    return state
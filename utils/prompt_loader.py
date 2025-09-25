import os
from typing import Dict, Optional

class PromptLoader:
    """프롬프트 파일을 로드하고 관리하는 유틸리티"""
    
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            # 기본 prompts 디렉토리 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompts_dir = os.path.join(os.path.dirname(current_dir), "prompts")
        else:
            self.prompts_dir = prompts_dir
            
        self._prompt_cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        프롬프트 파일을 로드하고 캐싱
        
        Args:
            prompt_name: 프롬프트 파일 이름 (확장자 제외)
            
        Returns:
            프롬프트 텍스트
        """
        # 캐시 확인
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        
        # 파일 경로 생성
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.py")
        
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
        
        # 프롬프트 파일에서 PROMPT_TEMPLATE 변수 로드
        try:
            # 파일을 모듈로 실행하여 변수 가져오기
            namespace = {}
            with open(prompt_file, 'r', encoding='utf-8') as f:
                exec(f.read(), namespace)
            
            # 프롬프트 템플릿 변수 찾기 (우선순위: _PROMPT_TEMPLATE > _PROMPT)
            prompt_template = None
            
            # 먼저 _PROMPT_TEMPLATE을 찾기 (응답 생성용)
            for var_name, value in namespace.items():
                if var_name.endswith('_PROMPT_TEMPLATE'):
                    prompt_template = value
                    break
            
            # _PROMPT_TEMPLATE이 없으면 _PROMPT 찾기
            if prompt_template is None:
                for var_name, value in namespace.items():
                    if var_name.endswith('_PROMPT'):
                        prompt_template = value
                        break
            
            if prompt_template is None:
                raise ValueError(f"프롬프트 템플릿을 찾을 수 없습니다: {prompt_file}")
            
            # 캐시에 저장
            self._prompt_cache[prompt_name] = prompt_template
            return prompt_template
            
        except Exception as e:
            raise RuntimeError(f"프롬프트 로드 중 오류 발생: {e}")
    
    def get_task_prompt(self, task_type: str) -> str:
        """
        태스크 타입에 따른 프롬프트 로드
        
        Args:
            task_type: C1, C2, C3, C4 중 하나
            
        Returns:
            해당 태스크의 프롬프트 템플릿
        """
        prompt_name = f"{task_type.lower()}_prompt"
        return self.load_prompt(prompt_name)
    
    def get_classifier_prompt(self) -> str:
        """태스크 분류기 프롬프트 로드"""
        return self.load_prompt("task_classifier_prompt")
    
    def get_reflection_prompt(self) -> str:
        """리플렉션 프롬프트 로드"""
        return self.load_prompt("reflection_prompt")
    
    def clear_cache(self):
        """프롬프트 캐시 초기화"""
        self._prompt_cache.clear()
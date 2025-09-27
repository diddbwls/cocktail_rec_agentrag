import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from utils.llm_model import get_llm
from utils.config import LLM_MODEL

load_dotenv()

class OpenAIClient:
    """
    Wrapper class for LLM operations
    This class now uses the new LLM system that supports multiple models
    """
    def __init__(self, model: Optional[str] = None, temperature: float = 0):
        """
        Initialize OpenAIClient
        
        Args:
            model: Model name override (if None, uses config.LLM_MODEL)
            temperature: Temperature for generation
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature
        self.llm = get_llm(self.model)
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, response_format: Optional[str] = None) -> str:
        """
        Generate text using LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            response_format: "json" for JSON response format (note: not all models support this)
            
        Returns:
            Generated text response
        """
        try:
            # Add JSON instruction to prompt if requested
            if response_format == "json":
                prompt = f"{prompt}\n\nRespond only with valid JSON."
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response
            
        except Exception as e:
            print(f"LLM API 호출 오류: {e}")
            raise e
    
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # JSON 블록 추출 (```json ... ``` 형식 처리)
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"응답: {response}")
            # 기본값 반환
            return {"error": "JSON 파싱 실패", "raw_response": response}
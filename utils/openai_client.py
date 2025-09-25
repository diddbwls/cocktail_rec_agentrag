import openai
from typing import Optional, List, Dict, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, response_format: Optional[str] = None) -> str:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            response_format: "json" for JSON response format
            
        Returns:
            Generated text response
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API 호출 오류: {e}")
            raise e
    
    def generate_with_system(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text with system prompt
        
        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API 호출 오류: {e}")
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
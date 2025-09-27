"""
LLM Model Factory for supporting multiple language models
Supports OpenAI GPT models and OpenRouter models (like Qwen)
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseLLM(ABC):
    """Base class for all LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: float = 0) -> str:
        """Generate text response"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        """Chat completion with message history"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT models implementation"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: float = 0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(messages, temperature, max_tokens)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise e


class OpenRouterLLM(BaseLLM):
    """OpenRouter models implementation (Qwen, etc.)"""
    
    def __init__(self, model_name: str = "qwen/qwen2.5-vl-72b-instruct:free"):
        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("QWEN_API_KEY")
        )
        self.model_name = model_name
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: float = 0) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(messages, temperature, max_tokens)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            raise e


class QwenVLLLM(BaseLLM):
    """Local Qwen VL models implementation using Transformers"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen VL model and processor"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            print(f"🔄 Loading {self.model_name}...")
            
            # Load model and processor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            print(f"✅ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            raise e
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: float = 0) -> str:
        """Generate text response (text-only)"""
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Prepare input for text-only generation
            inputs = self.processor(text=full_prompt, return_tensors="pt")
            
            # Move to device
            import torch
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or 512,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if full_prompt in response:
                response = response.replace(full_prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Qwen VL generation error: {e}")
            raise e
    
    def generate_with_image(self, prompt: str, image_path: str, 
                           max_tokens: Optional[int] = None, temperature: float = 0) -> str:
        """Generate text response with image input"""
        try:
            from PIL import Image
            import torch
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare input with image and text
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            import torch
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or 512,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Qwen VL image generation error: {e}")
            raise e
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        """Chat completion (converts to single prompt)"""
        # Convert messages to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n".join(prompt_parts) + "\nAssistant:"
        return self.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)


class LLMFactory:
    """Factory class to create appropriate LLM instances"""
    
    @staticmethod
    def create(model_name: str) -> BaseLLM:
        """
        Create LLM instance based on model name
        
        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "Qwen/Qwen2.5-VL-3B-Instruct")
            
        Returns:
            LLM instance
        """
        # Local Qwen VL models (Transformers)
        if model_name.startswith("Qwen/Qwen2.5-VL"):
            return QwenVLLLM(model_name)
        
        # OpenAI GPT models
        elif model_name.startswith(("gpt-", "text-davinci")):
            return OpenAILLM(model_name)
        
        # OpenRouter models (remote Qwen, Claude, etc.)
        elif "/" in model_name and not model_name.startswith("Qwen/Qwen2.5-VL"):
            return OpenRouterLLM(model_name)
        elif model_name.startswith("qwen"):
            return OpenRouterLLM(model_name)
        
        # Default to OpenAI
        else:
            print(f"Unknown model type: {model_name}, defaulting to OpenAI")
            return OpenAILLM(model_name)


# Convenience function for backward compatibility
def get_llm(model_name: Optional[str] = None) -> BaseLLM:
    """
    Get LLM instance with default from config
    
    Args:
        model_name: Optional model name override
        
    Returns:
        LLM instance
    """
    if model_name is None:
        from utils.config import LLM_MODEL
        model_name = LLM_MODEL
    
    return LLMFactory.create(model_name)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from util import config

class QwenModel:
    def __init__(self):
        self.device = config.HF_DEVICE

        print(f"Loading model: {config.HF_MODEL_NAME} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.HF_MODEL_NAME,
            torch_dtype=getattr(torch, config.HF_TORCH_DTYPE),
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(config.HF_MODEL_NAME)
        self.model.eval()
        print("Model loaded ✅")

    def generate(self, prompt: str, max_length: int = None) -> str:
        max_length = max_length or config.MAX_GENERATION_LENGTH
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        # 모든 텐서를 모델 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_length)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return output_text[0]
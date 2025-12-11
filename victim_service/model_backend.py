
"""
Model Backend with 15+ Real HuggingFace Models
"""

import os
import time
import math
import random
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TimingObfuscator:
    def __init__(self, strategy: str = "none", param: float = 0.0):
        self.strategy = strategy
        self.param = param
        
    def obfuscate(self, start_time: float) -> None:
        if self.strategy == "none":
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.strategy == "random":
            time.sleep(random.uniform(0, self.param) / 1000.0)
        elif self.strategy == "bucket":
            target_ms = math.ceil(elapsed_ms / self.param) * self.param
            remaining = target_ms - elapsed_ms
            if remaining > 0:
                time.sleep(remaining / 1000.0)
        elif self.strategy == "constant":
            remaining = self.param - elapsed_ms
            if remaining > 0:
                time.sleep(remaining / 1000.0)


class RealModelBackend:
    """15+ Real HuggingFace Models"""
    
    MODEL_REGISTRY = {
        # GPT-2 Family
        "distilgpt2": ("distilgpt2", "causal", 82),
        "gpt2": ("gpt2", "causal", 124),
        "gpt2-medium": ("gpt2-medium", "causal", 355),
        "gpt2-large": ("gpt2-large", "causal", 774),
        
        # OPT Family
        "opt-125m": ("facebook/opt-125m", "causal", 125),
        "opt-350m": ("facebook/opt-350m", "causal", 350),
        
        # Pythia Family
        "pythia-70m": ("EleutherAI/pythia-70m", "causal", 70),
        "pythia-160m": ("EleutherAI/pythia-160m", "causal", 160),
        "pythia-410m": ("EleutherAI/pythia-410m", "causal", 410),
        
        # GPT-Neo
        "gpt-neo-125m": ("EleutherAI/gpt-neo-125M", "causal", 125),
        
        # BLOOM
        "bloom-560m": ("bigscience/bloom-560m", "causal", 560),
        
        # BERT Family
        "bert-tiny": ("prajjwal1/bert-tiny", "masked", 4),
        "bert-base": ("bert-base-uncased", "masked", 110),
        "distilbert": ("distilbert-base-uncased", "masked", 66),
        
        # Other
        "albert-base": ("albert-base-v2", "masked", 12),
        "t5-small": ("t5-small", "seq2seq", 60),
    }
    
    def __init__(self, model_name: str):
        if model_name not in self.MODEL_REGISTRY:
            available = ", ".join(sorted(self.MODEL_REGISTRY.keys()))
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        self.model_name = model_name
        self.hf_model_id, self.model_type, self.params_m = self.MODEL_REGISTRY[model_name]
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def _ensure_loaded(self):
        if self._loaded:
            return
            
        logger.info(f"Loading model: {self.hf_model_id} ({self.params_m}M params)")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
        
        torch.set_num_threads(4)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        elif self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.hf_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.hf_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        
        self.model.eval()
        self._loaded = True
        logger.info(f"Model {self.model_name} loaded")
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        self._ensure_loaded()
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            if self.model_type == "causal":
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif self.model_type == "seq2seq":
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = self.model(**inputs)
                return f"[{self.model_name}] Processed {inputs['input_ids'].shape[1]} tokens"


class ModelBackend:
    def __init__(self, model_name: str, use_real: bool = True,
                 obfuscation_strategy: str = "none", obfuscation_param: float = 0.0):
        self.model_name = model_name
        self.use_real = use_real
        self.obfuscator = TimingObfuscator(obfuscation_strategy, obfuscation_param)
        self._backend = RealModelBackend(model_name)
    
    def generate(self, prompt: str, **kwargs) -> tuple:
        start = time.perf_counter()
        output = self._backend.generate(prompt, **kwargs)
        actual_elapsed = (time.perf_counter() - start) * 1000
        self.obfuscator.obfuscate(start)
        obfuscated_elapsed = (time.perf_counter() - start) * 1000
        return output, actual_elapsed, obfuscated_elapsed
    
    @property
    def is_real(self) -> bool:
        return True
    
    @classmethod
    def list_models(cls) -> Dict[str, Any]:
        return {name: {"params_m": info[2], "type": info[1]} 
                for name, info in RealModelBackend.MODEL_REGISTRY.items()}

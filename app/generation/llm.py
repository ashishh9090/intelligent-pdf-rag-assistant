import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests

from app.core.config import settings
from app.core.logging import logger


class BaseLLM(ABC):
    """Abstract Base Class for LLM providers to ensure modularity and pluggability."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generates a text completion for the given prompt and optional system instructions."""
        pass


class LocalHuggingFaceLLM(BaseLLM):
    """Local Hugging Face CausalLM pipeline (e.g., Qwen2.5-0.5B-Instruct)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS

        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    torch.set_num_threads(1)
                    logger.info(f"Loading local Hugging Face LLM: {self.model_name}...")
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    self._model.eval()
                    logger.info(f"Local LLM '{self.model_name}' successfully loaded.")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._load_model()
        import torch

        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template if supported by tokenizer
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            formatted_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            sys_part = f"System: {system_prompt}\n\n" if system_prompt else ""
            formatted_input = f"{sys_part}User: {prompt}\n\nAssistant:"

        inputs = self._tokenizer(formatted_input, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temp > 0.0:
                gen_kwargs["temperature"] = max(temp, 0.05)
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][input_length:]
        response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response_text


class HuggingFaceAPILLM(BaseLLM):
    """Client for Hugging Face Inference API / Serverless Endpoints."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_key = api_key or settings.HUGGINGFACE_API_KEY
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY must be set to use HuggingFaceAPILLM.")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "return_full_text": False,
            },
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return str(data)


class MockLLM(BaseLLM):
    """
    High-fidelity deterministic Mock/Extractive LLM for instant automated testing,
    offline execution, and zero-hallucination validation.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        lower_prompt = prompt.lower()

        # Handle Query Rewriting prompts cleanly
        if "given the following conversation history and a follow-up question" in lower_prompt:
            if "follow-up question:" in lower_prompt:
                fu_part = prompt.split("Follow-up Question:")[1].split("Standalone")[0].strip()
                return f"{fu_part} for Acme Corp"
            return prompt

        if "answer: i couldn't find this information in the uploaded documents" in lower_prompt:
            return "I couldn't find this information in the uploaded documents."

        # Check if context is provided in prompt
        if "retrieved document context:" in lower_prompt:
            parts = prompt.split("Retrieved Document Context:")
            if len(parts) > 1:
                context_part = parts[1].split("Question:")[0].strip()
                question_part = parts[1].split("Question:")[1] if "Question:" in parts[1] else ""
                
                # Check for negative / absent knowledge
                if not context_part or "no relevant context" in context_part.lower():
                    return "I couldn't find this information in the uploaded documents."

                stopwords = {
                    "what", "which", "where", "there", "about", "could", "would",
                    "their", "these", "those", "according", "reports", "report",
                    "documents", "document", "stated", "mentioned", "fiscal"
                }
                words = [w.strip("?,.:'\"") for w in question_part.lower().split() if len(w) > 3 and w not in stopwords]
                matches = [w for w in words if w in context_part.lower()]
                
                # Refuse if key query entities are not grounded in retrieved context
                if words and (len(matches) / len(words)) < 0.35:
                    return "I couldn't find this information in the uploaded documents."

                # Synthesize informative grounded excerpt
                lines = [line.strip() for line in context_part.split("\n") if line.strip() and not line.startswith("[Source")]
                content_snippet = " ".join(lines[:2])
                return f"Based on the provided documents: {content_snippet}"

        return "I couldn't find this information in the uploaded documents."


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """Factory to instantiate the configured LLM provider."""
    selected = provider or settings.LLM_PROVIDER
    logger.info(f"Instantiating LLM provider: '{selected}'")

    if selected == "hf_api":
        return HuggingFaceAPILLM()
    elif selected == "mock":
        return MockLLM()
    else:
        # Default to local HuggingFace
        return LocalHuggingFaceLLM()

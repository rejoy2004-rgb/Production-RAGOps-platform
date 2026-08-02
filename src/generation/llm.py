#Communication with LLMs (OpenRouter, Google Gemini, Anthropic Claude) with retry and fallback logic
import time
import httpx
from typing import List, Optional
from src.config.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("llm")

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

# List of active free models on OpenRouter to cascade through if rate limits are hit
FREE_MODEL_FALLBACKS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free"
]

class LLMService:
    """Manages text generation using OpenRouter or direct Google Gemini API with exponential backoff retries and model cascading."""
    
    # Class-level dictionary to track rate limit cooldowns: {model_name: timestamp_when_cooldown_ends}
    _cooldowns = {}
    
    def __init__(self):
        self.openrouter_key = settings.openrouter_api_key
        self.gemini_key = settings.gemini_api_key
        self.anthropic_key = settings.anthropic_api_key

    def _call_openrouter(self, prompt: str, system_prompt: Optional[str] = None, model: str = settings.default_model) -> str:
        """Calls OpenRouter. Cascades through fallback models if primary model is rate-limited or deprecated."""
        if not self.openrouter_key:
            logger.warning("OPENROUTER_API_KEY is not set. LLM calls will fail.")
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter completions.")
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ragops-platform",
            "X-Title": "RAGOps Platform"
        }
        
        # Prepare list of models to try
        models_to_try = [model]
        if ":free" in model:
            for fb in FREE_MODEL_FALLBACKS:
                if fb != model and fb not in models_to_try:
                    models_to_try.append(fb)
        
        # Filter models that are currently in cooldown
        now = time.time()
        active_models = []
        for m in models_to_try:
            cooldown_until = LLMService._cooldowns.get(m, 0.0)
            if now < cooldown_until:
                logger.info(f"Model {m} is in rate-limit cooldown for another {int(cooldown_until - now)}s. Skipping...")
                continue
            active_models.append(m)
            
        # If all requested models are in cooldown, clear/ignore cooldowns to avoid failing
        if not active_models:
            logger.warning("All fallback models are in cooldown. Resetting cooldowns and retrying all.")
            active_models = models_to_try
            LLMService._cooldowns.clear()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for current_model in active_models:
            logger.info(f"Attempting OpenRouter completion with model: {current_model}")
            
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.0,
            }
            
            max_retries = 2
            backoff_factor = 3.0
            
            for attempt in range(max_retries + 1):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(url, headers=headers, json=payload)
                        
                        if response.status_code == 429:
                            # Put model in cooldown for 120 seconds
                            LLMService._cooldowns[current_model] = time.time() + 120.0
                            
                            if attempt == max_retries:
                                logger.warning(f"Rate limit exceeded for {current_model}. Trying next fallback model...")
                                last_error = "Rate limit exceeded (429)"
                                break
                                
                            # Respect Retry-After header if present
                            retry_after = response.headers.get("Retry-After")
                            try:
                                sleep_time = float(retry_after) if retry_after else (backoff_factor ** attempt)
                            except ValueError:
                                sleep_time = backoff_factor ** attempt
                                
                            logger.warning(f"Rate limit (429) hit for {current_model}. Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                            
                        if response.status_code == 404:
                            logger.warning(f"Model {current_model} not found (404). Trying next fallback model...")
                            last_error = "Model not found (404)"
                            break
                            
                        response.raise_for_status()
                        result = response.json()
                        answer = result["choices"][0]["message"]["content"]
                        # Success, clear cooldown if any
                        LLMService._cooldowns.pop(current_model, None)
                        return answer.strip()
                        
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Error calling {current_model} after retries: {e}")
                        last_error = str(e)
                        break
                        
                    sleep_time = backoff_factor ** attempt
                    logger.warning(f"Transient error: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    
        # If all models in the cascade failed
        raise ValueError(
            f"OpenRouter API call failed across all fallback models. Last error: {last_error}. "
            "Please check your network or try again later."
        )

    def _call_direct_gemini(self, prompt: str, system_prompt: Optional[str] = None, model: str = "gemini-2.5-flash") -> str:
        """Calls Google Gemini API directly via HTTP with retry handling for rate limits and network errors."""
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY is required for direct Gemini completions.")
            
        # Clean model name if passed from openrouter mapping
        model_name = model.split("/")[-1] if "/" in model else model
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        
        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": full_text}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            }
        }
        
        max_retries = 3
        backoff_factor = 2.0
        
        for attempt in range(max_retries + 1):
            logger.info(f"Calling Gemini Direct model {model_name} (attempt {attempt+1}/{max_retries+1})...")
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    
                    if response.status_code == 429:
                        if attempt == max_retries:
                            raise ValueError("Gemini API rate limit exceeded (429). Please try again later.")
                        sleep_time = backoff_factor ** attempt
                        logger.warning(f"Gemini Rate limit (429) hit. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                        
                    response.raise_for_status()
                    result = response.json()
                    answer = result["candidates"][0]["content"]["parts"][0]["text"]
                    return answer.strip()
                    
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                    
                if attempt == max_retries:
                    logger.error(f"Direct Gemini API call failed after {max_retries} retries: {e}")
                    raise e
                    
                sleep_time = backoff_factor ** attempt
                logger.warning(f"Transient error: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        raise ValueError("Failed to get response from Gemini API.")

    def _call_direct_claude(self, prompt: str, system_prompt: Optional[str] = None, model: str = "claude-3-5-sonnet-20240620") -> str:
        """Calls Anthropic Claude API directly via HTTP with retry handling for rate limits and network errors."""
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY is required for direct Anthropic Claude completions.")
            
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": messages,
            "temperature": 0.0
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        max_retries = 3
        backoff_factor = 2.0
        
        for attempt in range(max_retries + 1):
            logger.info(f"Calling Anthropic Direct model {model} (attempt {attempt+1}/{max_retries+1})...")
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    
                    if response.status_code == 429:
                        if attempt == max_retries:
                            raise ValueError("Anthropic API rate limit exceeded (429). Please try again later.")
                        sleep_time = backoff_factor ** attempt
                        logger.warning(f"Anthropic Rate limit (429) hit. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                        
                    response.raise_for_status()
                    result = response.json()
                    answer = result["content"][0]["text"]
                    return answer.strip()
                    
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                    
                if attempt == max_retries:
                    logger.error(f"Direct Anthropic Claude API call failed after {max_retries} retries: {e}")
                    raise e
                    
                sleep_time = backoff_factor ** attempt
                logger.warning(f"Transient error: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        raise ValueError("Failed to get response from Anthropic Claude API.")

    @traceable(name="LLMService.call_llm", run_type="llm")
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Dispatches LLM calls to the appropriate service (Gemini direct, Claude direct, or OpenRouter) with fallback."""
        import os
        import re
        if os.getenv("EVAL_MOCK") == "true" or os.getenv("MOCK_LLM") == "true":
            logger.info("Mock LLM mode is active. Returning simulated response.")
            sys_prompt_lower = (system_prompt or "").lower()
            if "query decomposition" in sys_prompt_lower:
                match = re.search(r'Question to decompose:\s*"(.*?)"', prompt)
                if match:
                    return match.group(1)
                return prompt
            elif "faithfulness" in sys_prompt_lower:
                return '{\n  "reasoning": "Mocked faithfulness check: answer matches retrieved context.",\n  "score": 1.0\n}'
            elif "context precision" in sys_prompt_lower:
                return '{\n  "reasoning": "Mocked context precision check: chunks are relevant.",\n  "score": 1.0\n}'
            elif "context recall" in sys_prompt_lower:
                return '{\n  "reasoning": "Mocked context recall check: retrieved context contains ground truth.",\n  "score": 1.0\n}'
            else:
                match = re.search(r'Question:\s*(.*?)\s*\n\nAnswer', prompt, re.DOTALL)
                q_text = match.group(1).strip() if match else "query"
                return f"Mocked generated answer for: {q_text} based on the retrieved context."

        model_to_use = model or settings.default_model
        
        # Route based on model prefix and API key availability
        if "gemini" in model_to_use.lower() and self.gemini_key:
            try:
                return self._call_direct_gemini(prompt, system_prompt, model_to_use)
            except Exception as e:
                logger.warning(f"Direct Gemini call failed, trying OpenRouter fallback... Error: {e}")
                if self.openrouter_key:
                    return self._call_openrouter(prompt, system_prompt, model_to_use)
                raise e
        elif "claude" in model_to_use.lower() and self.anthropic_key:
            try:
                return self._call_direct_claude(prompt, system_prompt, model_to_use)
            except Exception as e:
                logger.warning(f"Direct Claude call failed, trying OpenRouter fallback... Error: {e}")
                if self.openrouter_key:
                    return self._call_openrouter(prompt, system_prompt, model_to_use)
                raise e
        else:
            return self._call_openrouter(prompt, system_prompt, model_to_use)

    @traceable(name="LLMService.generate_answer", run_type="chain")
    def generate_answer(self, question: str, retrieved_context: str, model: Optional[str] = None) -> str:
        """Formats the query context and generates final answer using the specified model."""
        prompt = f"Context:\n{retrieved_context}\n\nQuestion:\n{question}\n\nAnswer using only the provided context."
        system_prompt = "You are a helpful assistant. You must answer the user's question using ONLY the provided context. If the answer cannot be found in the context, say 'I cannot find the answer in the provided context.'"
        
        return self.call_llm(prompt=prompt, system_prompt=system_prompt, model=model)

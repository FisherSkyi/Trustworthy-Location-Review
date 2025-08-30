import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import time

class LLMReviewClassifier:
    """LLM-based review classification using a local transformer model."""

    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        print(f"Loaded local model: {model_name}")

    def create_classification_prompt(self, review_text: str) -> list:
        system = ("You are a strict content moderator. Respond ONLY with a JSON object with keys "
                  "is_advertisement, is_irrelevant, is_rant_without_visit, reasoning.")
        user = f'''
                Analyze this review and respond ONLY with JSON:
                Review: "{review_text}"

                Schema:
                {{
                "is_advertisement": true/false,
                "is_irrelevant": true/false,
                "is_rant_without_visit": true/false,
                "reasoning": "short explanation"
                }}
                '''.strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def classify_review_with_prompt(self, review_text: str, idx: int = None, total: int = None) -> dict:
        """Classify one review and print progress logs"""
        try:
            if idx is not None and total is not None:
                print(f"Classifying review {idx+1}/{total} ...", flush=True)

            start = time.time()
            
            messages = self.create_classification_prompt(review_text)
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response part
            assistant_response = response_text.split("<|im_start|>assistant")[1].replace("<|im_end|>", "").strip()

            duration = time.time() - start
            print(f"Done review {idx+1}/{total} in {duration:.1f}s", flush=True)

            # Clean and parse JSON
            json_text = re.sub(r"^```json|```$", "", assistant_response, flags=re.I | re.M).strip()
            return json.loads(json_text)
        except Exception as e:
            print(f"Error on review {idx+1 if idx is not None else ''}: {e}", flush=True)
            return {
                "is_advertisement": False,
                "is_irrelevant": False,
                "is_rant_without_visit": False,
                "reasoning": f"error: {e}"
            }

    def batch_classify(self, reviews):
        results = []
        total = len(reviews)
        print(f"Starting batch classification of {total} reviews")
        for i, r in enumerate(reviews):
            result = self.classify_review_with_prompt(r, idx=i, total=total)
            results.append(result)

            # Print a heartbeat every 10 reviews
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{total} reviews so far", flush=True)
        print("Finished all reviews")
        return results


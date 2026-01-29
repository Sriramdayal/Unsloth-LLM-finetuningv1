import unsloth
from unsloth import FastLanguageModel
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_inference_model
from src.inference import generate_response

def main():
    # Example usage for LeetCoder style inference
    # Note: Adjust model names as needed
    BASE_MODEL = "unsloth/qwen2.5-0.5b-unsloth-bnb-4bit"
    LORA_MODEL = "black279/Qwen_LeetCoder"
    
    print(f"Loading Base Model: {BASE_MODEL} and LoRA: {LORA_MODEL}...")
    try:
        model, tokenizer = load_inference_model(BASE_MODEL, LORA_MODEL)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure you have trained the model or have access to the HuggingFace model.")
        return

    prompt = "Extract the product information:\n<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div>"
    print(f"Prompt: {prompt}")
    
    print("Generating Response...")
    response = generate_response(model, tokenizer, prompt)
    
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()

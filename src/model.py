import unsloth
from unsloth import FastLanguageModel
import torch
from .config import ModelConfig
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(config: ModelConfig):
    """
    Loads the base model with Unsloth optimization.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        # dtype=None, # Auto detection
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer

def add_lora_adapters(model, config: ModelConfig):
    """
    Adds LoRA adapters to the model.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=config.random_state,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def load_inference_model(base_model_name: str, adapter_model_name: str):
    """
    Loads a model for inference using standard transformers + peft.
    This mimics the logic in Qwen_LeetCoder.ipynb which uses PeftModel directly.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 # or autodetect
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_model_name
    )
    model.eval()
    
    return model, tokenizer

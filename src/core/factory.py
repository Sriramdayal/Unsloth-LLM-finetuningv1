import unsloth
from unsloth import FastLanguageModel
import logging
import torch
from ..config import ModelConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Thread-safe factory for loading and configuring Unsloth models and adapters.
    """
    
    @staticmethod
    def create_model_and_tokenizer(config: ModelConfig):
        """
        Loads the base model with specified configuration.
        """
        logger.info(f"Factory: Loading model {config.model_name_or_path}...")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = config.model_name_or_path,
                max_seq_length = config.max_seq_length,
                dtype = None, # Auto-detect
                load_in_4bit = config.load_in_4bit,
                device_map = "auto",
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Factory: Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    @staticmethod
    def apply_lora(model, config: ModelConfig):
        """
        Applies PEFT/LoRA adapters using Unsloth's optimized method.
        """
        logger.info("Factory: Patching model with LoRA adapters...")
        
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r = config.lora_r,
                target_modules = config.target_modules,
                lora_alpha = config.lora_alpha,
                lora_dropout = config.lora_dropout,
                bias = "none", # Optimized for Unsloth
                use_gradient_checkpointing = "unsloth",
                random_state = config.random_state,
                use_rslora = False,
                loftq_config = None,
            )
            return model
        except Exception as e:
            logger.error(f"Factory: Failed to apply LoRA: {e}")
            raise RuntimeError(f"LoRA application failed: {e}")

    @staticmethod
    def prepare_for_inference(model):
        """
        Switches model to optimized inference mode.
        """
        FastLanguageModel.for_inference(model)
        model.eval()
        return model

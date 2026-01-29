import logging
import torch
from .factory import ModelFactory
from ..config import ModelConfig

logger = logging.getLogger(__name__)

class ModelRunner:
    """
    High-level API for model loading, interaction, and lifecycle management.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_training_ready = False

    def setup_for_training(self):
        """Prepares model and tokenizer for SFT."""
        self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)
        self.model = ModelFactory.apply_lora(self.model, self.config)
        self.is_training_ready = True
        return self.model, self.tokenizer

    def setup_for_inference(self, adapter_path: str = None):
        """Loads model for inference, optionally with a specific adapter."""
        # Note: If adapter_path is provided, we might need a custom loader logic
        # For now, we assume loading the base model + default config lora
        if not self.model:
            self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)
            self.model = ModelFactory.apply_lora(self.model, self.config)
        
        self.model = ModelFactory.prepare_for_inference(self.model)
        return self.model, self.tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7):
        """Generates text from a prompt with chat template support."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call setup_for_inference() first.")
        
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

import unsloth
from unsloth import FastLanguageModel
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, TrainConfig
from src.data import load_movie_dataset
from src.model import load_model, add_lora_adapters
from src.train import train_model

def main():
    print("Initializing Configuration...")
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    print(f"Loading Dataset: {train_config.dataset_name}...")
    dataset = load_movie_dataset(train_config)
    
    print(f"Loading Model: {model_config.model_name}...")
    model, tokenizer = load_model(model_config)
    
    print("Adding LoRA Adapters...")
    model = add_lora_adapters(model, model_config)
    
    print("Starting Training...")
    trainer, stats = train_model(model, tokenizer, dataset, train_config, model_config)
    
    print("Training Complete!")
    print(stats)
    
    # Save the model
    print(f"Saving model to {train_config.output_dir}/final_merged_model_4bit...")
    model.save_pretrained_merged(
        f"{train_config.output_dir}/final_merged_model_4bit",
        tokenizer,
        save_method="merged_4bit",
    )
    
if __name__ == "__main__":
    main()

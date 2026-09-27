"""Minimal end-to-end fine-tuning run.

Usage:
    python examples/quickstart.py
"""

from src import DataProcessor, ModelConfig, ModelRunner, TrainConfig, train_model


def main():
    # 1. Configuration
    model_config = ModelConfig(
        model_name_or_path="unsloth/mistral-7b-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    train_config = TrainConfig(
        dataset_name="yahma/alpaca-cleaned",
        dataset_num_samples=100,  # small sample for a smoke test
        output_dir="outputs/quickstart",
    )

    # 2. Load model + apply LoRA (backend auto-selected for your OS/GPU)
    runner = ModelRunner(model_config)
    model, tokenizer = runner.setup_for_training()

    # 3. Load + format dataset
    processor = DataProcessor(model_config, train_config, tokenizer)
    processor.load_dataset()
    dataset = processor.format_and_tokenize()

    # 4. Train
    stats, output_path = train_model(model, tokenizer, dataset, train_config, model_config)
    print(f"Training complete. Model saved to: {output_path} (stats: {stats})")


if __name__ == "__main__":
    main()

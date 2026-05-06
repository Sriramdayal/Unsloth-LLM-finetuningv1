"""
Standalone training script example.
Demonstrates programmatic usage of the unsloth-finetuning library.
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
        dataset_num_samples=100,  # Small sample for test
        output_dir="outputs/test_run",
    )

    # 2. Setup Model & Runner
    runner = ModelRunner(model_config)
    model, tokenizer = runner.setup_for_training()

    # 3. Process Data
    processor = DataProcessor(model_config, train_config, tokenizer)
    processor.load_dataset()
    dataset = processor.format_and_tokenize()

    # 4. Start Training
    stats, output_path = train_model(model, tokenizer, dataset, train_config, model_config)
    print(f"Training complete. Model saved to: {output_path}")


if __name__ == "__main__":
    main()

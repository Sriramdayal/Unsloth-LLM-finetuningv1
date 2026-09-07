"""Run the DataProcessor ETL on a sample dataset and preview the output.

Usage:
    python examples/data_etl.py
"""

from src import DataProcessor, ModelConfig, TrainConfig


def main():
    model_config = ModelConfig(model_name_or_path="dummy")
    train_config = TrainConfig(
        dataset_name="yahma/alpaca-cleaned",
        dataset_num_samples=10,
        dataset_style="alpaca",
    )

    # tokenizer=None is supported for preview: EOS token degrades to ""
    processor = DataProcessor(model_config, train_config, tokenizer=None)
    processor.load_dataset()
    dataset = processor.format_and_tokenize()

    print(f"columns={dataset.column_names} rows={len(dataset)}")
    for row in dataset[:3]:
        print("-" * 60)
        print(row["text"][:500])


if __name__ == "__main__":
    main()

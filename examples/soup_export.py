"""Export a trained model to GGUF via the Soup CLI (hybrid integration).

Usage:
    pip install -e ".[soup]"
    python examples/soup_export.py ./outputs/quickstart --quant q4_k_m
"""

import sys

from src import ModelConfig, TrainConfig
from src.soup import SoupClient, SoupConfig


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/quickstart"
    quant = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--quant" else "q4_k_m"

    # Optional: generate a soup.yaml from this repo's configs
    soup_config = SoupConfig.from_model_train_configs(
        ModelConfig(model_name_or_path=model_path),
        TrainConfig(dataset_name="yahma/alpaca-cleaned", output_dir=model_path),
    )
    soup_config.to_yaml("soup.yaml")
    print("Wrote soup.yaml")

    client = SoupClient()
    if not client.is_available():
        raise SystemExit(
            "The 'soup' binary was not found. Install with: pip install -e \".[soup]\""
        )
    out = client.export_gguf(model_path, quant=quant)
    print(f"Exported GGUF: {out}")


if __name__ == "__main__":
    main()

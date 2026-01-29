import unsloth
from unsloth import FastLanguageModel
from src import ModelConfig, ModelRunner

def main():
    # 1. Configuration
    model_config = ModelConfig(
        model_name_or_path="unsloth/mistral-7b-bnb-4bit",
    )

    # 2. Setup Runner
    runner = ModelRunner(model_config)
    runner.setup_for_inference()

    # 3. Generate
    prompt = "Tell me a joke about programming."
    response = runner.generate(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()

import os

from smolagents import CodeAgent, InferenceClientModel, tool


@tool
def suggest_model(use_case: str, max_vram_gb: float) -> str:
    """Suggests an appropriate open-source model for fine-tuning based on the use case and VRAM.

    Args:
        use_case: The primary use case for the model, e.g., 'coding', 'math', 'general', 'medical'.
        max_vram_gb: The maximum available VRAM in gigabytes (e.g., 8, 16, 24, 80).
    """
    if max_vram_gb <= 8:
        if use_case.lower() == "coding":
            return "unsloth/Qwen2.5-Coder-1.5B-Instruct or unsloth/Llama-3.2-1B-Instruct"
        else:
            return "unsloth/Llama-3.2-1B-Instruct or unsloth/Llama-3.2-3B-Instruct"
    elif max_vram_gb <= 16:
        if use_case.lower() == "coding":
            return "unsloth/Qwen2.5-Coder-7B-Instruct"
        else:
            return "unsloth/meta-llama-3.1-8b-instruct or unsloth/gemma-2-9b"
    else:
        if use_case.lower() == "coding":
            return "unsloth/Qwen2.5-Coder-32B-Instruct or unsloth/DeepSeek-R1-Distill-Llama-8B"
        else:
            return "unsloth/meta-llama-3.3-70b-instruct (with 4-bit) or unsloth/Mistral-Nemo"


@tool
def suggest_finetuning_parameters(model_name: str, target_modules_type: str = "all") -> str:
    """Suggests LoRA and Training parameters for fine-tuning.

    Args:
        model_name: The name of the model being fine-tuned.
        target_modules_type: The type of target modules to tune. Can be 'all', 'attention', or 'mlp'. Default is 'all'.
    """
    params = f"""
    Suggested Parameters for {model_name}:
    - LoRA Rank (r): 16
    - LoRA Alpha: 16 (or 32 for some models)
    - Target Modules ({target_modules_type}): 
      { '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' if target_modules_type == 'all' else '["q_proj", "k_proj", "v_proj", "o_proj"]'}
    - Learning Rate: 2e-4
    - Batch Size: 2 (gradient accumulation steps: 4)
    - Optimizer: paged_adamw_8bit
    - Weight Decay: 0.01
    """
    return params


def create_finetuning_agent(hf_token: str = None) -> CodeAgent:
    """Creates a CodeAgent capable of suggesting models, parameters, and writing code for LLM fine-tuning."""
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "A Hugging Face token is required to use the InferenceClientModel. Please set the HF_TOKEN environment variable."
        )

    # Using a capable open-source model via HF inference API
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token)

    agent = CodeAgent(
        tools=[suggest_model, suggest_finetuning_parameters],
        model=model,
        additional_authorized_imports=["torch", "transformers", "trl", "peft", "datasets"],
        description="I am an expert LLM fine-tuning assistant. I can suggest models, configuration parameters, and write Unsloth fine-tuning code.",
    )
    return agent


if __name__ == "__main__":
    print("Welcome to the LLM Fine-tuning Agent powered by smolagents!")
    print("Make sure you have set the HF_TOKEN environment variable.")
    print(
        "Example Query: 'I have 16GB of VRAM and want to train a coding model. What model and params should I use?'"
    )

    try:
        agent = create_finetuning_agent()
        while True:
            query = input("\nWhat would you like help with? (type 'exit' to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
            try:
                response = agent.run(query)
                print("\n=== Agent Response ===")
                print(response)
                print("======================")
            except Exception as e:
                print(f"Error during agent execution: {e}")
    except ValueError as e:
        print(f"Startup Error: {e}")

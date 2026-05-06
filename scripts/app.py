"""
"No-Code" Fine-Tuning Studio (Gradio App).
State-Aware GUI for interactive fine-tuning.
"""

import gc
import logging
import os
import sys
import threading

import gradio as gr
import pandas as pd
import torch
from transformers import TrainerCallback

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig, TrainConfig
from src.core.factory import ModelFactory
from src.data import DataProcessor
from src.train import train_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Global State ---
class AppState:
    model = None
    tokenizer = None
    dataset = None
    processor = None
    is_training = False
    log_history: list[str] = []


state = AppState()


# --- Custom Callback for Real-Time Logs ---
class GradioLogCallback(TrainerCallback):
    def __init__(self, log_queue: list):
        self.log_queue = log_queue

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_str = f"Step {state.global_step}: Loss: {logs.get('loss', 'N/A')}\n"
            self.log_queue.append(log_str)


# --- Functions ---


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    if state.model is not None and state.model != "MOCK_MODEL":
        del state.model
        state.model = None
    if state.tokenizer is not None:
        del state.tokenizer
        state.tokenizer = None
    state.dataset = None
    state.processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return "Memory Cleared. Model unloaded."


def preview_data(dataset_name: str, num_samples: int):
    """Loads and previews the dataset."""
    try:
        model_cfg = ModelConfig(model_name_or_path="dummy")
        train_cfg = TrainConfig(
            dataset_name=dataset_name,
            dataset_num_samples=int(num_samples) if num_samples else None,
        )

        processor = DataProcessor(model_cfg, train_cfg, tokenizer=None)
        processor.load_dataset()

        # Preview raw data as DataFrame (first 5 rows)
        df = pd.DataFrame(processor.raw_dataset[:5])
        return df, "Dataset Loaded Successfully"
    except Exception as e:
        logger.exception("Failed to preview dataset")
        return pd.DataFrame(), f"Error: {e}"


def load_model_and_tokenize(model_name, load_4bit, r, alpha, dataset_name, dataset_style, use_mock):
    """Loads the model and prepares the dataset."""
    try:
        status_msg = ""

        # Load Model
        if state.model is None and not use_mock:
            yield "Loading Model (this may take time)...", pd.DataFrame()

            cfg = ModelConfig(
                model_name_or_path=model_name,
                lora_r=int(r),
                lora_alpha=int(alpha),
                load_in_4bit=load_4bit,
            )

            model, tokenizer = ModelFactory.create_model_and_tokenizer(cfg)
            model = ModelFactory.apply_lora(model, cfg)

            state.model = model
            state.tokenizer = tokenizer
            status_msg += f"Model {model_name} loaded.\n"

        elif use_mock:
            from transformers import AutoTokenizer

            yield "Loading Mock Tokenizer (gpt2)...", pd.DataFrame()
            state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            state.tokenizer.pad_token = state.tokenizer.eos_token
            state.model = "MOCK_MODEL"  # Sentinel value
            status_msg += "Mock Model Enabled (CPU Mode).\n"

        else:
            status_msg += "Model already loaded (Clear Memory to change).\n"

        # Process Data
        model_cfg = ModelConfig(model_name_or_path=model_name, use_mock=use_mock)
        train_cfg = TrainConfig(dataset_name=dataset_name)

        state.processor = DataProcessor(model_cfg, train_cfg, state.tokenizer)
        state.processor.load_dataset()

        # Format using user-selected style
        dataset = state.processor.format_and_tokenize(style=dataset_style)
        state.dataset = dataset
        status_msg += f"Dataset {dataset_name} processed using {dataset_style} style.\n"

        # Preview formatted data (first 3 rows)
        formatted_preview = pd.DataFrame(dataset[:3])
        yield status_msg, formatted_preview

    except Exception as e:
        logger.exception("Failed to load model or process data")
        yield f"Error: {e}", pd.DataFrame()


def train_wrapper(batch_size, lr, epochs, output_dir, use_mock):
    """Wrapper to run training in a thread-safe way for Gradio."""
    if state.model is None or state.dataset is None:
        return "Error: Model or Dataset not loaded."

    AppState.log_history = []

    train_cfg = TrainConfig(
        dataset_name="loaded_in_memory",
        output_dir=output_dir,
        batch_size=int(batch_size),
        learning_rate=float(lr),
        num_train_epochs=float(epochs),
    )

    model_cfg = ModelConfig(model_name_or_path="loaded", use_mock=use_mock)

    # Custom Callback for live log streaming
    class ListLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log_entry = f"Step {state.global_step}: Loss {logs.get('loss', 'N/A')}\n"
                AppState.log_history.append(log_entry)

    try:
        stats, path = train_model(
            state.model,
            state.tokenizer,
            state.dataset,
            train_cfg,
            model_cfg,
            callbacks=[ListLogCallback()],
        )
        return f"Training Complete! Model saved to {path}. Stats: {stats}"
    except Exception as e:
        logger.exception("Training failed")
        return f"Training Failed: {e}"


def stream_logs():
    """Generator to stream logs to the UI."""
    return "".join(AppState.log_history)


def ai_assistant_chat(message, history):
    """Heuristic AI Assistant to recommend hyperparameters and datasets."""
    msg = message.lower()
    response = "🤖 **AI Tuning Agent:**\n\n"
    if "code" in msg or "programming" in msg:
        response += "For coding tasks, I highly recommend using **`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`** as your base model, and a dataset like **`m-a-p/CodeFeedback-Filtered-Instruction`**. Set your LoRA Rank to `32` to capture the complex syntax, and use a learning rate of `3e-4`."
    elif "chat" in msg or "conversation" in msg:
        response += "For conversational AI, **`unsloth/llama-3-8b-bnb-4bit`** is excellent. Use **`HuggingFaceH4/no_robots`** with the **ChatML** format. Keep the learning rate around `2e-4` with Rank `16`."
    elif "medical" in msg or "science" in msg:
        response += "For medical/scientific domains, you need high factual accuracy. Try **`unsloth/mistral-7b-v0.3-bnb-4bit`**. Use a high LoRA Rank of `64` to memorize the dense terminology, and train for at least `3 Epochs`."
    elif "agent" in msg or "tool" in msg or "function" in msg:
        response += "To train an AI Agent that can use tools (Function Calling), use **`unsloth/llama-3-8b-bnb-4bit`**. Set the Dataset Style to **`agent`** in the Data Setup tab. The dataset should be in ShareGPT format with `<tool_call>` fields."
    else:
        response += "I can help you pick the best hyperparameters and datasets! Tell me what kind of data you're training on (e.g., 'coding', 'chatbots', 'agents', or 'medical text')."
    return response


# --- UI Setup ---
custom_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_400)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
)

custom_css = """
.gradio-container { max-width: 1200px !important; }
.header-title { text-align: center; padding: 1rem 0; font-weight: 900; background: linear-gradient(90deg, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.status-box { border-radius: 8px; border: 1px solid #4ade80; }
"""

with gr.Blocks(title="Unsloth Fine-Tuning Studio", theme=custom_theme, css=custom_css) as app:
    gr.Markdown("<h1 class='header-title'>🦥 Unsloth Fine-Tuning Studio</h1>")
    gr.Markdown(
        "<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Interactive, High-Performance LLM Training powered by Unsloth & Gradio</p>"
    )

    with gr.Row():
        status_box = gr.Textbox(
            label="System Status", interactive=False, lines=2, elem_classes="status-box"
        )
        clear_btn = gr.Button("🗑️ Clear GPU Memory", variant="stop", scale=0)

    clear_btn.click(fn=clear_memory, outputs=status_box)

    with gr.Tabs():
        # Tab 1: Data & Model
        with gr.TabItem("⚙️ 1. Model & Data Setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Model Configuration")
                    model_name = gr.Dropdown(
                        [
                            "unsloth/llama-3-8b-bnb-4bit",
                            "unsloth/mistral-7b-v0.3-bnb-4bit",
                            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                        ],
                        label="Model Preset",
                        allow_custom_value=True,
                        value="unsloth/llama-3-8b-bnb-4bit",
                        info="Select a pre-quantized Unsloth model or type any HuggingFace ID.",
                    )

                    with gr.Accordion("Advanced Model Settings", open=False):
                        load_4bit = gr.Checkbox(
                            label="Load in 4-bit (bitsandbytes)",
                            value=True,
                            info="Highly recommended to save VRAM.",
                        )
                        use_mock = gr.Checkbox(
                            label="Mock Mode (CPU Test)",
                            value=False,
                            info="Simulate training without GPU.",
                        )
                        lora_r = gr.Slider(
                            8,
                            256,
                            step=8,
                            value=16,
                            label="LoRA Rank (r)",
                            info="Higher rank = smarter, but slower and larger.",
                        )
                        lora_alpha = gr.Slider(
                            16,
                            512,
                            step=16,
                            value=16,
                            label="LoRA Alpha",
                            info="Scaling factor for LoRA weights.",
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Dataset Configuration")
                    dataset_name = gr.Textbox(
                        label="Hugging Face Dataset ID",
                        value="yahma/alpaca-cleaned",
                        info="Enter the dataset repository name.",
                    )
                    dataset_style = gr.Dropdown(
                        ["auto", "alpaca", "chat", "agent", "movie_recommender"],
                        label="Dataset Style (Prompt Format)",
                        value="auto",
                        info="Choose 'agent' to format Tool Calling data, or 'chat' for conversations.",
                    )
                    dataset_limit = gr.Number(
                        label="Limit Samples",
                        value=100,
                        info="Number of rows to load for preview/testing.",
                    )

                    with gr.Row():
                        preview_btn = gr.Button("👁️ Preview Raw Data", size="sm")
                        process_btn = gr.Button(
                            "🚀 Load Model & Tokenize", variant="primary", size="sm"
                        )

            with gr.Accordion("Data Previews", open=True):
                raw_preview = gr.Dataframe(
                    label="Raw Data Preview", headers=["Column 1", "Column 2"], max_height=200
                )
                formatted_preview = gr.Dataframe(
                    label="Formatted Data Preview (Tokenized Text)", max_height=200
                )

            # Wiring up the buttons
            preview_btn.click(
                preview_data,
                inputs=[dataset_name, dataset_limit],
                outputs=[raw_preview, status_box],
            )
            process_btn.click(
                load_model_and_tokenize,
                inputs=[
                    model_name,
                    load_4bit,
                    lora_r,
                    lora_alpha,
                    dataset_name,
                    dataset_style,
                    use_mock,
                ],
                outputs=[status_box, formatted_preview],
            )

        # Tab 2: Training Params
        with gr.TabItem("⚡ 2. Hyperparameters"):
            gr.Markdown("### 🎛️ Training Hyperparameters")
            with gr.Row():
                with gr.Column():
                    batch_size = gr.Slider(
                        1,
                        32,
                        step=1,
                        label="Batch Size",
                        value=2,
                        info="Samples per device. Reduce if Out-of-Memory.",
                    )
                    lr = gr.Number(
                        label="Learning Rate", value=2e-4, info="Step size for optimizer."
                    )
                with gr.Column():
                    epochs = gr.Slider(
                        0.1,
                        10,
                        step=0.1,
                        label="Epochs",
                        value=1.0,
                        info="Number of full passes over dataset.",
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="outputs",
                        info="Folder to save final checkpoints.",
                    )

        # Tab 3: Run & Monitor
        with gr.TabItem("📈 3. Train & Monitor"):
            gr.Markdown("### 🚀 Launch Training")
            with gr.Row():
                start_train_btn = gr.Button("🔥 Start Training Process", variant="primary", scale=2)

            with gr.Row():
                logs_output = gr.Textbox(
                    label="Live Training Logs (Auto-updating)",
                    lines=15,
                    max_lines=20,
                    elem_classes="console-log",
                )

            result_box = gr.Textbox(label="Final Status / Results", lines=2)

            # Polling for logs
            timer = gr.Timer(1)
            timer.tick(fn=stream_logs, inputs=None, outputs=logs_output)

            start_train_btn.click(
                fn=train_wrapper,
                inputs=[batch_size, lr, epochs, output_dir, use_mock],
                outputs=[result_box],
            )

        # Tab 4: AI Assistant
        with gr.TabItem("🤖 4. AI Tuning Agent"):
            gr.Markdown("### 💬 Ask the AI for Fine-Tuning Advice")
            gr.Markdown(
                "Not sure what LoRA Rank, dataset, or learning rate to use? Ask the AI Tuning Agent!"
            )
            gr.ChatInterface(
                fn=ai_assistant_chat,
                chatbot=gr.Chatbot(height=400, elem_classes="status-box"),
                examples=[
                    "I want to train a coding model",
                    "What settings for a medical assistant?",
                    "How to train an AI agent with tool calling?",
                ],
            )


def main():
    """Entry point for the Gradio app."""
    app.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()

import unsloth
from unsloth import FastLanguageModel
"""
"No-Code" Fine-Tuning Studio (Gradio App).
State-Aware GUI for interactive fine-tuning.
"""
import gradio as gr
import pandas as pd
import threading
import torch
import gc
from transformers import TrainerCallback

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ModelConfig, TrainConfig
from src.data import DataProcessor
from src.train import train_model

# --- Global State ---
class AppState:
    model = None
    tokenizer = None
    dataset = None
    processor = None
    is_training = False
    log_history = []

state = AppState()

# --- Custom Callback for Real-Time Logs ---
class GradioLogCallback(TrainerCallback):
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Format logs for display
            log_str = f"Step {state.global_step}: Loss: {logs.get('loss', 'N/A')}\n"
            self.log_queue.append(log_str)

# --- Functions ---

def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    if state.model:
        del state.model
        state.model = None
    if state.tokenizer:
        del state.tokenizer
        state.tokenizer = None
    torch.cuda.empty_cache()
    gc.collect()
    return "Memory Cleared. Model unloaded."

def preview_data(dataset_name, num_samples):
    """Loads and previews the dataset."""
    try:
        # Create dummy configs for loading
        model_cfg = ModelConfig(model_name_or_path="dummy") 
        train_cfg = TrainConfig(dataset_name=dataset_name, dataset_num_samples=int(num_samples) if num_samples else None)
        
        # We need a dummy tokenizer or load a real one. 
        # For previewing "raw" data, we don't strictly need one if we just look at columns, 
        # but DataProcessor expects one in init. 
        # Let's use a lightweight check or just suppress if not needed for get_preview.
        # Actually, let's just proceed.
        
        processor = DataProcessor(model_cfg, train_cfg, tokenizer=None)
        processor.load_dataset()
        df = processor.get_preview(n=5)
        return df, "Dataset Loaded Successfully"
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def load_model_and_tokenize(model_name, load_4bit, r, alpha, dataset_name, use_mock):
    """Loads the model and prepares the dataset."""
    try:
        status_msg = ""
        
        # Load Model
        if state.model is None and not use_mock:
            yield "Loading Model (this may take time)...", pd.DataFrame()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = 2048,
                dto = None,
                load_in_4bit = load_4bit,
            )
            
            model = FastLanguageModel.get_peft_model(
                model,
                r = int(r),
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha = int(alpha),
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
            )
            state.model = model
            state.tokenizer = tokenizer
            status_msg += f"Model {model_name} loaded.\n"
            
        elif use_mock:
             from transformers import AutoTokenizer
             yield "Loading Mock Tokenizer (gpt2)...", pd.DataFrame()
             state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
             state.tokenizer.pad_token = state.tokenizer.eos_token
             state.model = "MOCK_MODEL" # Sentinel value
             status_msg += "Mock Model Enabled (CPU Mode).\n"

        else:
            status_msg += "Model already loaded (Clear Memory to change).\n"

        # Process Data
        # Ensure model_config carries the mock flag for DataProcessor if needed, though mostly for train
        model_cfg = ModelConfig(model_name_or_path=model_name, use_mock=use_mock) 
        train_cfg = TrainConfig(dataset_name=dataset_name)
        
        state.processor = DataProcessor(model_cfg, train_cfg, state.tokenizer)
        state.processor.load_dataset()
        
        # Assume alpaca for now
        dataset = state.processor.format_and_tokenize(style="alpaca")
        state.dataset = dataset
        status_msg += f"Dataset {dataset_name} processed.\n"
        
        # Preview Formatted
        formatted_preview = pd.DataFrame(dataset[:3])
        
        yield status_msg, formatted_preview

    except Exception as e:
        yield f"Error: {str(e)}", pd.DataFrame()


def train_wrapper(batch_size, lr, epochs, output_dir, use_mock):
    """Wrapper to run training in a thread safe way for Gradio."""
    if state.model is None or state.dataset is None:
        return "Error: Model or Dataset not loaded."
    
    state.log_history = []
    
    train_cfg = TrainConfig(
        dataset_name="loaded_in_memory",
        output_dir=output_dir,
        batch_size=int(batch_size),
        learning_rate=float(lr),
        num_train_epochs=float(epochs)
    )
    
    model_cfg = ModelConfig(model_name_or_path="loaded", use_mock=use_mock)

    # Custom Callback handling
    class ListLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                 log_entry = f"Step {state.global_step}: Loss {logs.get('loss', 'N/A')}\n"
                 # Append to global log history for the UI to poll
                 AppState.log_history.append(log_entry)

    try:
        stats, path = train_model(
            state.model, 
            state.tokenizer, 
            state.dataset, 
            train_cfg, 
            model_cfg, 
            callbacks=[ListLogCallback()]
        )
        return f"Training Complete! Model saved to {path}. Stats: {stats}"
    except Exception as e:
        return f"Training Failed: {e}"

def stream_logs():
    """Generator to stream logs to the UI."""
    return "".join(AppState.log_history)

# --- UI Setup ---

with gr.Blocks(title="Unsloth Fine-Tuning Studio") as app:
    gr.Markdown("# 🦥 Unsloth Fine-Tuning Studio")
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear GPU Memory", variant="stop")
        status_box = gr.Textbox(label="System Status", interactive=False)

    clear_btn.click(fn=clear_memory, outputs=status_box)

    with gr.Tabs():
        # Tab 1: Data & Model
        with gr.TabItem("1. Data & Model"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        ["unsloth/llama-3-8b-bnb-4bit", "unsloth/mistral-7b-v0.3-bnb-4bit", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"], 
                        label="Model Preset", 
                        allow_custom_value=True,
                        value="unsloth/llama-3-8b-bnb-4bit"
                    )
                    with gr.Row():
                         load_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
                         use_mock = gr.Checkbox(label="Mock Mode (CPU Test)", value=False)
                    lora_r = gr.Slider(8, 256, value=16, label="LoRA Rank (r)")
                    lora_alpha = gr.Slider(16, 512, value=16, label="LoRA Alpha")
                
                with gr.Column():
                    dataset_name = gr.Textbox(label="Hugging Face Dataset ID", value="yahma/alpaca-cleaned")
                    dataset_limit = gr.Number(label="Limit Samples (for test)", value=100)
                    preview_btn = gr.Button("Load & Preview Raw Data")
            
            raw_preview = gr.Dataframe(label="Raw Data Preview", headers=["Column 1", "Column 2"])
            preview_btn.click(preview_data, inputs=[dataset_name, dataset_limit], outputs=[raw_preview, status_box])
            
            process_btn = gr.Button("Load Model & Process Data", variant="primary")
            formatted_preview = gr.Dataframe(label="Formatted Data Preview (Tokenized Text)")
            process_btn.click(load_model_and_tokenize, 
                              inputs=[model_name, load_4bit, lora_r, lora_alpha, dataset_name, use_mock], 
                              outputs=[status_box, formatted_preview])
        
        # Tab 2: Training Params
        with gr.TabItem("2. Training Params"):
            batch_size = gr.Number(label="Batch Size", value=2)
            lr = gr.Number(label="Learning Rate", value=2e-4)
            epochs = gr.Number(label="Epochs", value=1)
            output_dir = gr.Textbox(label="Output Directory", value="outputs")

        # Tab 3: Run & Monitor
        with gr.TabItem("3. Run & Monitor"):
            start_train_btn = gr.Button("🚀 Start Training", variant="primary")
            logs_output = gr.Textbox(label="Live Training Logs", lines=10, max_lines=20)
            result_box = gr.Textbox(label="Final Result")

            # Polling for logs
            timer = gr.Timer(1)
            timer.tick(fn=stream_logs, inputs=None, outputs=logs_output)
            
            start_train_btn.click(fn=train_wrapper, 
                                  inputs=[batch_size, lr, epochs, output_dir, use_mock], 
                                  outputs=[result_box])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)

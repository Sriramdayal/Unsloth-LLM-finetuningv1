
import unsloth
from unsloth import FastLanguageModel
import os
import ast
from datasets import load_dataset
from src import ModelConfig, TrainConfig, DataProcessor, train_model

def safe_parse(value):
    try:
        return [item["name"] for item in ast.literal_eval(value)]
    except:
        return []

def format_row(row):
    genres = ", ".join(safe_parse(row["genres"]))
    keywords = ", ".join(safe_parse(row["keywords"]))

    overview = row["overview"] if row["overview"] else "No overview available."
    tagline = row["tagline"] if row.get("tagline") else ""

    return {
        "text": f"""### Instruction:
Generate a persuasive movie recommendation for a user. Highlight why they should watch the movie using its genre, themes, storyline, and popularity.

### Movie Metadata:
Title: {row['title']}
Tagline: {tagline}
Overview: {overview}
Genres: {genres}
Keywords: {keywords}
Vote Average: {row['vote_average']}
Vote Count: {row['vote_count']}
Popularity: {row['popularity']}
Release Date: {row['release_date']}
Runtime: {row['runtime']} minutes

### Response:
Here’s why you might enjoy this movie:
"""
    }

def main():
    # 1. Configuration
    model_config = ModelConfig(
        model_name_or_path="unsloth/mistral-7b-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    train_config = TrainConfig(
        dataset_name="Mariodb/movie-recommender-dataset", # Used for reference, loaded manually below
        output_dir="outputs/movie_recommender",
        num_train_epochs=1,
    )

    # 2. Setup Model & Tokenizer
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_config.model_name_or_path,
        max_seq_length = model_config.max_seq_length,
        dtype = None,
        load_in_4bit = model_config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Custom Data Loading & Formatting
    print("Loading and formatting dataset...")
    ds = load_dataset(train_config.dataset_name)
    train_ds = ds["train"].map(format_row)
    
    # Filter only relevant columns to keep memory usage low
    required_columns = ["text"]
    train_ds = train_ds.remove_columns(
        [col for col in train_ds.column_names if col not in required_columns]
    )

    # 4. Process with DataProcessor (handles tokenization checks etc)
    processor = DataProcessor(model_config, train_config, tokenizer)
    
    # Inject our pre-formatted dataset
    processor.raw_dataset = train_ds
    
    # This will detect the "text" column and use it directly
    dataset = processor.format_and_tokenize()

    # 5. Start Training
    print("Starting training...")
    train_model(model, tokenizer, dataset, train_config, model_config)

if __name__ == "__main__":
    main()

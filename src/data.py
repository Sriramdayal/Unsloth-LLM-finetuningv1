from datasets import load_dataset, Dataset
import ast
from .config import TrainConfig

def safe_parse(value):
    try:
        return [item["name"] for item in ast.literal_eval(value)]
    except:
        return []

def format_movie_prompt(row):
    genres = ", ".join(safe_parse(row["genres"]))
    keywords = ", ".join(safe_parse(row["keywords"]))

    overview = row["overview"] if row["overview"] else "No overview available."
    tagline = row["tagline"] if row.get("tagline") else ""

    text = f"""### Instruction:
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
    return {"text": text}

def load_movie_dataset(config: TrainConfig) -> Dataset:
    ds = load_dataset(config.dataset_name)
    train_ds = ds["train"].map(format_movie_prompt)
    
    # Remove all columns except 'text'
    train_ds = train_ds.remove_columns(
       [col for col in train_ds.column_names if col != "text"]
    )
    return train_ds

import torch

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generates a response for a given prompt.
    """
    # Ensure model is in inference mode
    # For Unsloth generic models
    if hasattr(model, "for_inference"):
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    
    response = tokenizer.batch_decode(outputs)
    # Simple extraction might be needed depending on the chat template behavior
    # This returns the full text including prompt usually, but batch_decode behavior 
    # might vary. We'll return the raw list for now.
    return response[0]

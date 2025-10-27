# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


def load_messages(model_name, prompt):
    templates = {
        "gemini-1.5-pro": prompt,
        "o4-mini": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        
        # Models run locally
        "Deepseek-math-7b-rl": [
            {"role": "user", "content": prompt},
        ],
        "Qwen-2.5-math-7B": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "Mathstral-7B": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "OpenMath2-Llama3.1-8B": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "OREAL-7B": [
            {"role": "system", "content": "You are an expert mathematician with extensive experience in mathematical competitions."},
            {"role": "user",   "content": prompt}
        ],
    }
    return templates.get(model_name)


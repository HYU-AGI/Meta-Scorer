# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_local_model(args):
    logger = logging.getLogger(__name__)
    model_name = args.model_name
    model_id = args.model_id


    if model_name in ["Deepseek-math-7b-rl", 
                      "Qwen-2.5-math-7B", 
                      "Mathstral-7B", 
                      "OpenMath2-Llama3.1-8B", 
                      "OREAL-7B"]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=args.cache_dir
        )
   
    else:
        raise ValueError(f"Local model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return model, tokenizer


def generate_local_response(model, tokenizer, messages, args):
    logger = logging.getLogger(__name__)
    model_name = args.model_name
    model_id = args.model_id

    model.generation_config = GenerationConfig.from_pretrained(model_id)
    model.generation_config.output_attentions = True
    model.generation_config.return_dict_in_generate = True

    if model_name in ["Deepseek-math-7b-rl", "OpenMath2-Llama3.1-8B", 'Mathstral-7B']:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").to(model.device)
        token_count = input_ids["input_ids"].shape[1]
        if token_count > 2048:
            logger.info(f"Input token count {token_count} exceeds 2048. Generation skipped.")
            return "", None, None

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.generation_config.eos_token_id
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0]
        model.generation_config.pad_token_id = pad_id

        outputs = model.generate(
            input_ids["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=None if args.top_p == 0.0 else args.top_p,
            temperature=None if args.temperature == 0 else args.temperature,
        )

        generated_ids = outputs.sequences[0][ input_ids["input_ids"].shape[-1] : ]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return response

    elif model_name in ["Qwen-2.5-math-7B", "OREAL-7B"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        token_count = model_inputs.input_ids.shape[1]
        if token_count > 2048:
            logger.info(f"Input token count {token_count} exceeds 2048. Generation skipped.")
            return "", None, None

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_k=None if args.top_k == 0 else args.top_k,
            top_p=None if args.top_p == 0.0 else args.top_p,
            temperature=None if args.temperature == 0 else args.temperature,
        )

        response = tokenizer.batch_decode(
            generated_ids.sequences[:, len(model_inputs.input_ids[0]):],
            skip_special_tokens=True
        )[0]

        return response.strip()

    else:
        logger.error(f"Response generation for {args.model_name} is not implemented.")
        raise ValueError(f"Response generation for {args.model_name} is not implemented.")
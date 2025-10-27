# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


import logging
import sys
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI
from utils import load_api_keys


def load_api_model(args):
    logger = logging.getLogger(__name__)
    model_name = args.model_name
    model_id = args.model_id
    api_keys = load_api_keys()

    if model_name in ["gemini-1.5-pro"]:
        genai.configure(api_key=api_keys["GEMINI_API_KEY"])
        client = genai.GenerativeModel(model_id)
    elif model_name in ["o4-mini"]:
        client = OpenAI(api_key=api_keys["OPENAI_API_KEY"])
    else:
        logger.error(f"API model {model_name} is not supported.")
        raise ValueError(f"API model {model_name} is not supported.")

    logger.info(f"Loading model {model_name}...")
    return client


def generate_api_response(model_name, client, messages, args):
    logger = logging.getLogger(__name__)
    model_id = args.model_id

    if model_name in ["o4-mini"]:
        max_retries = 10  
        retry_count = 0
        while retry_count < max_retries:
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    max_completion_tokens=args.max_new_tokens,
                    messages=messages,
                )
                response = completion.choices[0].message.content
                break
            except ResourceExhausted as e:
                logger.error(
                    f"Resource exhausted. Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})"
                )
                time.sleep(5)
                retry_count += 1
        if retry_count == max_retries:
            logger.error(f"Max retries reached. Could not complete the request.")
            sys.exit(1)

    elif model_name in ["gemini-1.5-pro"]:
        generation_config = genai.GenerationConfig(
            max_output_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.generate_content(messages, generation_config=generation_config).text
                break
            except ResourceExhausted as e:
                logger.error(
                    f"Resource exhausted. Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})"
                )
                time.sleep(5)
                retry_count += 1
        if retry_count == max_retries:
            logger.error(f"Max retries reached. Could not complete the request.")
            sys.exit(1) 
            
    else:
        logger.error(f"Response generation for {model_name} is not implemented.")
        raise ValueError(f"Response generation for {model_name} is not implemented.")

    return response.strip()

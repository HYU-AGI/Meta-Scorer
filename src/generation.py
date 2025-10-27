# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath

import argparse
import logging
import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from logger import setup_logger
from models import ModelWrapper
from prompts import load_novel_solution_generation_prompt
from utils import load_json, save_json, floatify


model_version = {
    "Deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
    "Qwen-2.5-math-7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "Mathstral-7B": "mistralai/Mathstral-7b-v0.1",
    "OpenMath2-Llama3.1-8B": "nvidia/OpenMath2-Llama3.1-8B",
    "OREAL-7B": "internlm/OREAL-7B"
}



def main():
    parser = argparse.ArgumentParser(description="Run Step 1. Generation and Step 2. Feature Extraction")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="Mathstral-7B", help="The model used to generate novel solutions.")
    parser.add_argument("--dataset_name", type=str, default="CreativeMath", help="dataset name")
    parser.add_argument("--do_sample", type=bool, default=True, help="do sample - True or False")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature setting")
    parser.add_argument("--top_p", type=float, default=1.0, help="LLM top_p setting")
    parser.add_argument("--top_k", type=int, default=50, help="LLM top_k setting")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="maximum new generation tokens")
    parser.add_argument("--log_dir", type=str, default="logs", help="logging log dir")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging log level")
    parser.add_argument("--cache_dir", type=str, default="cache", help="model cache dir")
    parser.add_argument("--output_dir", type=str, default="output", help="output dir")
    parser.add_argument("--data_dir", type=str, default="data", help="dataset dir")

    args = parser.parse_args()
    

    log_dir = os.path.join(args.log_dir, args.dataset_name, 'generation')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")
    logger = setup_logger(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting the generation on {args.dataset_name} using {args.model_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    
    # 모델 불러오기
    args.model_id = model_version[args.model_name]
    model = ModelWrapper(args)

    # 데이터셋 불러오기
    data_file = os.path.join(args.data_dir, f"{args.dataset_name}.json")
    dataset = load_json(data_file)

    # 생성 결과 저장 경로 설정
    output_dir = os.path.join(args.output_dir, args.dataset_name, 'generation')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model_name}.json")

    results = []

    for idx, sample in enumerate(tqdm(dataset)):
        problem = sample["problem"]
        solutions = list(sample["solutions"].values())

        # 입력 프롬프트 생성
        prompt = load_novel_solution_generation_prompt(problem, solutions, 1)
        # 응답 생성
        response = model.generate_response(prompt)

        if not response:
            continue

        # 결과 저장
        results.append(
            {"competition_id": sample["competition_id"], 
             "problem_id": sample["problem_id"],
             "problem": sample["problem"],
             "solutions": sample["solutions"],
             "response": response,
            }
        )
        if idx % args.save_interval == 0:
            save_json(floatify(results), output_file)
            
    save_json(floatify(results), output_file)
    logger.info(f"All results saved to {os.path.abspath(output_file)}")
            
if __name__ == "__main__":
    main()
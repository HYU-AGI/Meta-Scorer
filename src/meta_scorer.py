import os
import torch
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from logger import setup_logger
from models import ModelWrapper
from prompts import (load_novel_solution_generation_prompt,
                     load_correctness_evaluation_prompt,
                     load_coarse_grained_novelty_evaluation_prompt)
from utils import load_json, save_json, floatify, extract_yes_no
from models import ModelWrapper
from metrics_utils import compute_meta_scores_for_text


model_version = {
    "Deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
    "Qwen-2.5-math-7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "Mathstral-7B": "mistralai/Mathstral-7b-v0.1",
    "OpenMath2-Llama3.1-8B": "nvidia/OpenMath2-Llama3.1-8B",
    "OREAL-7B": "internlm/OREAL-7B",
    # LLM Evaluators
    "gemini-1.5-pro": "models/gemini-1.5-pro-002",
    "o4-mini": "o4-mini-2025-04-16"
}

evaluators = ["gemini-1.5-pro", "o4-mini"]


def main():
    parser = argparse.ArgumentParser(description="Compute meta-scores for already generated responses.")
    parser.add_argument("--seed", type=int, default=42, help="random seed setting")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="Mathstral-7B", help="Model name for generation.")
    parser.add_argument("--dataset_name", type=str, default="CreativeMath", help="dataset name")
    parser.add_argument("--generation_res_dir", type=str, default="output", help="Generation results directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for meta-scores")
    parser.add_argument("--cache_dir", type=str, default="cache", help="model cache dir")
    parser.add_argument("--log_dir", type=str, default="logs", help="logging log dir")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging log level")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature setting")
    
    args = parser.parse_args()
    
    log_dir = os.path.join(args.log_dir, args.dataset_name, 'meta_scoring')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")
    logger = setup_logger(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting meta-score computation for already generated responses...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")


    # 모델 불러오기
    args.model_id = model_version[args.model_name]
    model = ModelWrapper(args)

    # 저장된 모델 응답 불러오기
    saved_res_path = os.path.join(args.generation_res_dir, args.dataset_name, 'generation', f'{args.model_name}.json')
    saved_data = load_json(saved_res_path)

    # meta scoring 결과 저장 경로 설정
    output_dir = os.path.join(args.output_dir, args.dataset_name, 'meta_scoring')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model_name}_meta_scores.json")


    ###################################################
    # 정량적 평가
    ###################################################
    mt_list = ["logit", "hidden", "attns"]
    for idx, entry in tqdm(enumerate(saved_data)):
        if "meta-scores" in entry:
            save_json(floatify(saved_data), output_file)
            continue

        problem = entry["problem"]
        solutions = list(entry["solutions"].values())
        response = entry["response"]
        
        # logit, hidden states, attention 값을 구하기 위한 입력 프롬프트 생성
        prompt = load_novel_solution_generation_prompt(problem, solutions, entry['k'])
        combined_text = prompt + "\n" + response
        # 정량적 평가
        # meta score 계산 (perplexity, window entroy, logit entropy, hidden score, attention score)
        combined_metrics_dict = compute_meta_scores_for_text(
            text=combined_text,
            model=model,
            mt_list=mt_list,
        )
        entry["meta-scores"] = combined_metrics_dict

        # 결과 저장
        if idx % args.save_interval == 0:
            save_json(floatify(saved_data), output_file)

    ###################################################
    # 정성적 평가
    ###################################################
    for model_name in evaluators:
        args.model_name = model_name
        args.model_id = model_version[model_name]
        model = ModelWrapper(args)
    
        for idx, entry in tqdm(enumerate(saved_data)):
            if model_name in entry["labeling"]["correctness"]:
                save_json(floatify(saved_data), output_file)
                continue

            problem = entry["problem"]
            solutions = list(entry["solutions"].values())
            new_solution = entry["response"]

            # LLM Evaluator가 정답 여부 판별
            prompt = load_correctness_evaluation_prompt(problem, solutions, new_solution)
            response = model.generate_response(prompt)
        
            decision = extract_yes_no(response)  # Return either "YES" or "NO"
            entry["labeling"]["correctness"][model_name] = decision

            if entry["labeling"]["correctness"][model_name] == "NO":
                entry["labeling"]["novelty"][model_name] = "NO"
                # 하나라도 오답이라고 판단했으면 Hallucinated 로 간주
                entry["label"] = "Hallucinated_Solution"
                saved_data[idx] = entry
            else:
                k = entry['k']
                prompt = load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
                response = model.generate_response(prompt)
                decision = extract_yes_no(response)  # Return either "YES" or "NO"
                entry["labeling"]["novelty"][model_name] = decision
                saved_data[idx] = entry
            
            if idx % args.save_interval == 0:
                save_json(floatify(saved_data), output_file)
        
        save_json(floatify(saved_data), output_file)
        logger.info(f"Labeling results saved to {os.path.abspath(output_file)} using LLM-Evaluator {model_name}")
    
    for idx, entry in enumerate(saved_data):
        if entry['label'] != "Hallucinated_Solution":
            novelty = entry["labeling"]["novelty"].values()
            yes_count = sum(1 for value in novelty if value == "YES")
            if yes_count == 0:
                entry["label"] = "Typical_Solution"
            else:
                # 창의적이라고 판단한 LLM Evaluator가 하나 이상일 경우
                entry["label"] = "Creative_Solution"

    save_json(floatify(saved_data), output_file)
    logger.info(f"All results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":

    main()

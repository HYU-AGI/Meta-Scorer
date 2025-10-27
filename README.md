# 정량적 방식과 정성적 방식을 모두 고려하는 범용적 활용 가능한 Meta Scorer


## ⚙️ Requirements
To install requirements:
```
pip install -r requirements.txt
```

### ✨ LLM Evaluator 사용을 위해 API Key를 ```API_KEYS.json```에 입력해주세요.
API_KEYS.json 예시
```
{
    "GEMINI_API_KEY": "YOUR_GEMINI_API_KEY",
    "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY"
}
```

## 💻 Running Meta-Scorer
### Step 1. LLM을 활용한 답변 생성
```
python src/generation.py --model_name "generation_model_name" --dataset_name "CreativeMath" --do_sample True --temperature 0.7 --top_p 1.0 --top_k 0.7 --max_new_tokens 1024
```

### Step 2. Meta-Scorer 실행
```
python src/meta_scorer.py --model_name "generation_model_name" --dataset_name "CreativeMath"
```

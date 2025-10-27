import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_roc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low

def get_roc_auc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low, fpr, tpr


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str = None,
    dtype=torch.float16,
    **kwargs
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=dtype,
        **kwargs
    )
    model.requires_grad_(False)
    if model.generation_config.temperature is None:
        model.generation_config.temperature = 1.0
    model.generation_config.do_sample = True

    tokenizer_name_or_path = model_name_or_path if tokenizer_name_or_path is None else tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
    return model, tokenizer


def get_model_vals(model, tok_in):
    """
    한 번의 forward로 logits, hidden states, attentions을 얻는다.
    (output_attentions=True, output_hidden_states=True)
    반환할 때 마지막 2개 레이어만 슬라이싱하여 반환함.
    """
    kwargs = {
        "input_ids": tok_in,
        "use_cache": False,
        "past_key_values": None,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    with torch.no_grad():
        output = model(**kwargs)
    
    return output.logits, output.hidden_states, output.attentions


##############################################################################
# 로짓 기반 메트릭 (PPL, Entropy 등)
##############################################################################

def perplexity(logits, tok_ins, tok_lens, min_k=None):
    softmax = torch.nn.Softmax(dim=-1)
    ppls = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i]
        pr = torch.log(softmax(logits[i]))[torch.arange(i1, i2) - 1, tok_ins[i][i1:i2]]
        if min_k is not None:
            pr = torch.topk(pr, k=int(min_k * len(pr)), largest=False).values
        ppls.append(torch.exp(-pr.mean()).item())
    return np.stack(ppls)

def logit_entropy(logits, tok_lens, top_k=None):
    softmax = torch.nn.Softmax(dim=-1)
    scores = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i]
        if top_k is None:
            l = softmax(logits[i][i1:i2])
            scores.append((-l * torch.log(l)).mean())
        else:
            l = logits[i][i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
            scores.append((-l * torch.log(l)).mean())
    return np.stack(scores)

def window_logit_entropy(logits, tok_lens, top_k=None, w=1):
    softmax = torch.nn.Softmax(dim=-1)
    scores = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i]
        if top_k is None:
            l = softmax(logits[i])[i1:i2]
        else:
            l = logits[i][i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
        windows = torch.max(
            (-l * torch.log(l)).mean(1).unfold(0, w, w).mean(1)
        )
        scores.append(windows.item())
    return np.stack(scores)


##############################################################################
# Hidden states SVD
##############################################################################

def get_svd_eval(hidden_acts, layer_num=15, tok_lens=[], use_toklens=True):
    """
    hidden_acts[i][layer_num] : shape (seq_len, hidden_dim)
    -> transpose -> shape (hidden_dim, seq_len)
    -> centered_svd_val
    """
    svd_scores = []
    for i in range(len(hidden_acts)):
        Z = hidden_acts[i][layer_num]
        if use_toklens and tok_lens[i]:
            i1, i2 = tok_lens[i]
            Z = Z[i1:i2, :]
        Z = torch.transpose(Z, 0, 1)  # (hidden_dim, seq_len)
        val = centered_svd_val(Z)
        svd_scores.append(val.item())
    return np.stack(svd_scores)

def centered_svd_val(Z, alpha=0.001):
    """
    Z: (hidden_dim, seq_len), dtype could be bfloat16, float16, etc.
    """
    dim = Z.shape[0]  # hidden_dim
    # J를 Z와 같은 dtype, device로 생성
    J = torch.eye(dim, device=Z.device, dtype=Z.dtype) \
        - (1/dim)*torch.ones(dim, dim, device=Z.device, dtype=Z.dtype)

    # Sigma = (Z^T * J * Z).  (Z^T: (seq_len, hidden_dim), but here carefully see shapes)
    # 여기선 Z가 (hidden_dim, seq_len) -> Z.t() = (seq_len, hidden_dim)
    # => (seq_len, hidden_dim) x (hidden_dim, hidden_dim) => (seq_len, hidden_dim)
    # => x Z => (seq_len, seq_len)
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)

    # alpha * I
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], device=Z.device, dtype=Z.dtype)

    # GPU bfloat16/float16 -> svdvals 불가능할 수 있으므로 float()
    Sigma = Sigma.float()

    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore


##############################################################################
# Attention 메트릭
##############################################################################
def get_attn_eig_prod(attns, layer_num=0, tok_lens=None, use_toklens=True):
    """
    attns: 마지막 2개 레이어의 어텐션을 담은 리스트. 
           각 요소는 해당 레이어의 head 텐서 리스트 (각 텐서 shape: (seq_len, seq_len)) 입니다.
    layer_num: 계산에 사용할 레이어 인덱스 (0 또는 1).
    tok_lens: (i1, i2) 형태의 튜플로, 토큰 범위를 지정합니다. (예: (start, end))
              만약 None이면 전체 시퀀스를 사용합니다.
    use_toklens: True인 경우 tok_lens가 제공되면 해당 범위 내에서 계산합니다.
    
    반환값: 지정한 레이어에 대해 계산된 평균 log diagonal 스코어 (float)
    """
    head_list = attns[layer_num]  # 지정한 레이어의 head 리스트
    eigscore_sum = 0.0
    for head in head_list:
        Sigma = head
        if use_toklens and tok_lens is not None:
            i1, i2 = tok_lens
            Sigma = Sigma[i1:i2, i1:i2]
        eigscore_sum += torch.log(torch.diagonal(Sigma, 0)).mean()
    avg_eigscore = eigscore_sum / len(head_list)
    return avg_eigscore.item()


##############################################################################
# meta-scorer
##############################################################################
def compute_meta_scores_for_text(text, model, mt_list):
    tokenizer = model.tokenizer
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.model.device)

    logits, hidden_states, attentions = get_model_vals(model.model, input_ids)
    logits_cpu = logits[0].cpu()

    # 마지막 2개 hidden states만 사용
    hidden_acts_cpu = []
    for layer_h in hidden_states[-2:]:
        hidden_acts_cpu.append(layer_h[0].cpu())

    # 마지막 2개 attention layer만 사용
    attentions = attentions[-2:]
    attns_cpu = []
    for layer_a in attentions:
        heads_list = []
        for head_idx in range(layer_a.shape[1]):
            heads_list.append(layer_a[0, head_idx].cpu())
        attns_cpu.append(heads_list)

    logits_list = [logits_cpu]
    hidden_acts = [tuple(hidden_acts_cpu)]
    
    seq_len = logits_cpu.shape[0]
    tok_ins = [torch.arange(seq_len)]
    # tok_lens를 리스트가 아니라 단일 튜플로 생성하거나, 아래처럼 get_attn_eig_prod 호출 시 tok_lens[0]을 사용
    tok_lens = [(0, seq_len)]

    metric_dict = {}
    for mt in mt_list:
        if mt == "logit":
            ppl_val = perplexity(logits_list, tok_ins, tok_lens)[0]
            metric_dict["perplexity"] = ppl_val
            w_ent = window_logit_entropy(logits_list, tok_lens, w=1)[0]
            metric_dict["window_entropy"] = w_ent
            l_ent = logit_entropy(logits_list, tok_lens, top_k=50)[0]
            metric_dict["logit_entropy"] = l_ent

        elif mt == "hidden":
            for idx, _ in enumerate(hidden_acts[0]):
                key_svd = f"Hly{idx+1}"
                svd_val = get_svd_eval(hidden_acts, idx, tok_lens, True)[0]
                metric_dict[key_svd] = svd_val

        elif mt == "attns":
            for layer_num in range(len(attns_cpu)):
                key_eig = f"Attn{layer_num+1}"
                # tok_lens[0]를 넘겨서 단일 튜플을 사용하고, 함수가 float를 반환하므로 [0] 인덱싱 제거
                val_eig = get_attn_eig_prod(attns_cpu, layer_num, tok_lens[0], True)
                metric_dict[key_eig] = val_eig
        else:
            raise ValueError(f"Invalid method type: {mt}")

    return {k: float(v) for k, v in metric_dict.items()}

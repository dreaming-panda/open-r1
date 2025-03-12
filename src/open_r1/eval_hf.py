from datasets import load_dataset
import os
import argparse
import torch
import torch.multiprocessing as mp
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from datetime import datetime
from lm_eval.models.utils import (
    stop_sequences_criteria,
)

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

import random
import numpy as np
from open_r1.SparseQwen import Qwen2ForCausalLM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 适用于多 GPU 训练

    # 让 cuDNN 采用确定性算法（保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker(rank, world_size, args, shared_list):
    
    llm = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2", sliding_window=True).to(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm.eval()

    dataset = load_dataset("math-ai/amc23", split="test")
    
    total = len(dataset)

    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)  
    subset = dataset.select(list(range(start, end)))
    
    expr_gold_metric = multilingual_extractive_match_metric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(ExprExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    )

    for i in range(args.trial):
        set_seed(42 + i)
        for data in tqdm(subset, desc=f"Process {rank}", position=rank):
                golds = [data["answer"]]
                target = Doc(query=data["question"],choices=golds, gold_index=0)
                coversation = [
                    {"role": "user", "content":MATH_QUERY_TEMPLATE.format(Question=data["question"])}
                ]
                inputs = tokenizer.apply_chat_template(conversation=coversation, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(rank)

                
                stopping_criteria = stop_sequences_criteria(tokenizer, ['<|endoftext|>'], inputs.shape[1], inputs.shape[0])
                
                output = llm.generate(input_ids=inputs,
                                stopping_criteria=stopping_criteria, 
                                temperature=0.6,
                                top_p = 0.95, 
                                max_new_tokens=30768)
                
                predictions = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                result = expr_gold_metric.compute(golds=golds,predictions=[predictions],formatted_doc=target)
                shared_list.append(
                    {   
                        "id": data["id"],
                        "score": result["extractive_match"] * 100,
                        "prediction": [predictions],
                        "gold_index": 0,
                        "choices": golds,
                        "query": data["question"]
                    }
                )
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name')
    parser.add_argument('--task', type=str, default="niah_single_3", help='Task identifier')
    parser.add_argument('--output_dir', type=str, default="results", help='Output directory')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--trial', type=int, default=1, help='Number of trials')
    args = parser.parse_args()

    

    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(args.output_dir, f"{args.task}_{current_time}.jsonl")

    # Use mp.Manager to create a shared list for process-safe communication
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_list = manager.list()

    # Launch multiple processes using mp.spawn; each process handles a subset of the dataset
    world_size = args.nproc
    mp.spawn(worker, args=(world_size, args, shared_list), nprocs=world_size, join=True)

    # Gather results and compute global accuracy (optional)
    total_accuracy = 0.0
    count = 0
    results = []
    for item in shared_list:
            results.append(item)
            total_accuracy += item['score']
            count += 1

    global_summary = {
        'task': args.task,
        'accuracy': total_accuracy / count if count > 0 else 0,
        'total_example': count
    }
    results.insert(0, global_summary)

    # Save the aggregated results to a single JSONLines file
    with jsonlines.open(output_file, "w") as f:
        f.write_all(results)

    print("Results saved to", output_file)

if __name__ == "__main__":
    main()
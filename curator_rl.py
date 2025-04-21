import datasets
import copy

# 加载数据集
dapo = datasets.load_dataset("open-r1/DAPO-Math-17k-Processed", "en")

# 转换函数
def transform_example(example):
    example["problem"] = copy.deepcopy(example["prompt"])
    example["answer"] = "$\\boxed{{{}}}$".format(copy.deepcopy(example["solution"]))
    example["solution"] = "$\\boxed{{{}}}$".format(copy.deepcopy(example["solution"]))
    return example

# 执行映射
dapo = dapo.map(transform_example)

# 推送至 huggingface hub（确保登录过 huggingface-cli）
dapo.push_to_hub("ZMC2019/DAPO-Math-17k", private=True)

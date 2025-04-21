import datasets
import copy
dapo = datasets.load_dataset("open-r1/DAPO-Math-17k-Processed", "en")

def transform_example(example):
    example["problem"] = copy.deepcopy(example["prompt"])
    example["answer"] = copy.deepcopy(example["solution"])
    return example

dapo = dapo.map(transform_example)
dapo.push_to_hub("ZMC2019/DAPO-Math-17k", private=True)
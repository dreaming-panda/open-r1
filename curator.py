from datasets import load_dataset, Features, Value
from datasets import DatasetDict
from datasets import concatenate_datasets
features = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string"),
            "info": {
                "source": Value("string"),
                "reference_answer": Value("string"),
                "test_case": Value("string"),
                "think_content": Value("string"),
                "answer_content": Value("string")
            }
        }
    ]
})

data_900k = load_dataset('a-m-team/AM-DeepSeek-R1-Distilled-1.4M', 'am_0.9M', features=features)
#data_500k = load_dataset('a-m-team/AM-DeepSeek-R1-Distilled-1.4M', 'am_0.5M', features=features)

#merged_train = concatenate_datasets([data_900k['train'], data_500k['train']])

merged_dataset = DatasetDict({'train': data_900k['train']})
merged_dataset.push_to_hub("ZMC2019/AM0.9M")
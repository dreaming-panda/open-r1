# SPDX-License-Identifier: Apache-2.0

import argparse

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser

from transformers import AutoTokenizer

def create_test_prompts(tokenizer) -> list[tuple[str, SamplingParams]]:
    messages_list = [
        [{"role": "user", "content": "What is the meaning of life?"}],
    ]

    sampling_params_list = [
        SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=512),
    ]

    prompts = []
    for messages, params in zip(messages_list, sampling_params_list):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append((prompt, params))

    return prompts


def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output.outputs[0].text)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    test_prompts = create_test_prompts(tokenizer)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
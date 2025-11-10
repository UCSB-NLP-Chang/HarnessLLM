import argparse
import copy
import importlib.util
import json
import os
import sys

import datasets
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a specified model and prompts."
    )
    # required positional arguments
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path"
    )
    parser.add_argument(
        "prompt_file",
        type=str,
        help="Path to the file containing prompts"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Name of the dataset to use for generation"
    )

    # optional arguments with defaults
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the generated output"
    )
    parser.add_argument(
        "--start_ind",
        type=int,
        default=0,
        help="Index at which to start generation"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port number for the server (default: %(default)s)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: %(default)s)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32000,
        help="Maximum length of generated text (default: %(default)s)"
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=1,
        help="Number of samples to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=500,
        help="Number of samples to test (default: %(default)s)"
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=16,
        help="Number of processes to use for generation (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)"
    )

    return parser.parse_args()


def call_api(
    messages,
    max_tokens=None,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop=None,
    presence_penalty=0,
    seed=42,
    port=8000,
):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
    models = client.models.list()
    model = models.data[0].id
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        n=n,
        stop=stop,
        seed=seed,
    )

    return result


def get_qwen_question_template_answer(question):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question['question_content']}\n\n"
    if question['starter_code']:
        prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{question['starter_code']}\n```"
    else:
        prompt += "Read the inputs from stdin to solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt


def get_messages(module_name):
    file_path = f"prompts/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MESSAGE_TEMPLATE


def generate_testcase(data_path, outfile, prompt_file, model, start_ind, end_ind, num_proc=1, temperature=0.6, max_length=32000, num_sample=1, port=8000, seed=42):
    messages = get_messages(prompt_file)
    tokenizer = AutoTokenizer.from_pretrained(model)

    data = datasets.load_dataset(data_path)["train"]
    if start_ind is not None:
        print(f"====================== Total # of data: {len(data)} ======================")
        print(f"====================== Processing {start_ind}-{end_ind} ======================")
        data = data.select(range(start_ind, min(end_ind, len(data))))
    first_uid = data[0]["uid"]
    result_field = "output_text" if "codegen" in prompt_file else "testcase"
    print(f"Save to ====================== {outfile} ======================")
    print(f"Max length: {max_length}")
    print(f"Num sample: {num_sample}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Write to field: {result_field}")

    def get_testcase(item):
        messages_ = copy.deepcopy(messages)
        if "codegen" in prompt_file:
            assert len(messages_) == 1
            messages_[-1]["content"] = get_qwen_question_template_answer(item)
        else:
            gt = json.loads(item["reward_model"]["ground_truth"])
            testing_code = gt["bad_code"]['code'] if isinstance(gt["bad_code"], dict) else gt["bad_code"]
            if isinstance(messages_, dict):
                messages_ = messages_[gt.get("code_type", "input/output")]
            assert len(messages_) == 1
            problem = item["description"].strip()
            messages_[-1]["content"] = messages_[-1]["content"].format(description=problem, testing_code=testing_code.strip())
        try:
            if item["uid"] == first_uid:
                print(tokenizer.apply_chat_template(
                    messages_,
                    tokenize=False,
                    add_generation_prompt=True,
                ))
            response = call_api(
                messages=messages_,
                max_tokens=max_length,
                temperature=temperature,
                top_p=0.95,
                presence_penalty=1.5,
                n=num_sample,
                seed=seed,
                port=port,
            )
            item[result_field] = []
            for choice in response.choices:
                res = choice.message.content.strip()
                item[result_field].append(res)
                if item["uid"] == first_uid:
                    print(res)
        except Exception as e:
            print(e)
            item[result_field] = [""]
        return item

    data = data.map(get_testcase, num_proc=num_proc, batched=False, load_from_cache_file=False)
    data.to_parquet(outfile)


if __name__ == "__main__":
    args = parse_args()
    end_ind = args.start_ind + args.num_data
    outdir = f"{args.output_dir}/{args.data_path.split('/')[-1]}"
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/{args.model.split('/')[-1]}_{args.prompt_file}_{args.start_ind}_{args.seed}.parquet"
    generate_testcase(
        args.data_path,
        outfile,
        args.prompt_file,
        args.model,
        args.start_ind,
        end_ind,
        num_proc=args.num_process,
        temperature=args.temperature,
        max_length=args.max_length,
        num_sample=args.num_sample,
        port=args.port,
        seed=args.seed
    )
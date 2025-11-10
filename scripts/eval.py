import argparse
import importlib.util
import os
import pprint
import sys
from collections import Counter

import datasets
import numpy as np
import ray
from ray.exceptions import GetTimeoutError, RayTaskError, WorkerCrashedError
from tqdm import tqdm

import custom_verl.testcase_judge as testcase_judge
from custom_verl.reward_utils import RewardType
from custom_verl.testcase_judge import compute_score, get_feedback


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

    # optional arguments with defaults
    parser.add_argument(
        "--mode",
        type=str,
        default="eval_testcase", # Options: eval_testcase, get_feedback
        help="Mode of operation: eval_testcase for evaluation, get_feedback for feedback generation (default: %(default)s)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory containing the input data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Name of the dataset to use for evaluation"
    )
    parser.add_argument(
        "--num_testcases",
        type=int,
        default=20,
        help="Number of test cases to evaluate (default: %(default)s)"
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the test cases (default: %(default)s)"
    )
    parser.add_argument(
        "--num_generators",
        type=int,
        default=5,
        help="Number of input generators to use (default: %(default)s)"
    )
    parser.add_argument(
        "--aggregate_num",
        type=int,
        default=1,
        help="Number of test cases to aggregate (default: %(default)s)"
    )
    parser.add_argument(
        "--num_cases_per_generator",
        type=int,
        default=4,
        help="Number of test cases to generate per generator (default: %(default)s)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the generated output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)"
    )

    args = parser.parse_args()
    if not args.input_dir:
        assert args.data_path, "Either input_dir or data_path must be provided."
        args.input_dir = f"results/{args.data_path.split('/')[-1]}"
    if not args.output_dir:
        args.output_dir = args.input_dir
    return args


def get_messages(module_name):
    file_path = f"prompts/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MESSAGE_TEMPLATE

ray.init(namespace="evaluating", _temp_dir="/tmp/ray_session_code")

def aggregate_reawrd_types(r_types):
    if not any(rt in [RewardType.GoodInputFailGood, RewardType.GoodInputPassGoodPassBad, RewardType.PassGoodFailBad] for rt in r_types):
        # all inputs are bad
        if all(rt == RewardType.FormatError for rt in r_types):
            return RewardType.FormatError
        if any(rt == RewardType.FailGood for rt in r_types):
            return RewardType.FailGood
        return RewardType.PassGoodPassBad
    # some inputs are good
    if any(rt in [RewardType.GoodInputFailGood, RewardType.FailGood] for rt in r_types):
        return RewardType.GoodInputFailGood
    if any(rt == RewardType.PassGoodFailBad for rt in r_types):
        return RewardType.PassGoodFailBad
    return RewardType.GoodInputPassGoodPassBad


@ray.remote
def get_score_remote(item, num_testcases=20, test_harness=True, aggregate_num=1, num_repeats=1, num_generators=5):
    testcase_judge.GLOBAL_TIMEOUT = 20  # timeout for evaluation
    scores, r_types, fail_reasons, fail_indices, num_inputs = [], [], [], [], []
    if aggregate_num > 1:
        assert len(item["testcase"]) % aggregate_num == 0
        cache_scores, cache_r_types = [], []
    for ind, tc in enumerate(item["testcase"]):
        score, r_type, fail_reason, num_input = compute_score(
            tc,
            item["reward_model"]["ground_truth"],
            num_testcases=num_testcases,
            test_harness=test_harness,
            num_repeats=num_repeats,
            timeout=20,
            seed=42,  # Use a fixed seed for reproducibility
            num_generators=num_generators,
        )
        assert isinstance(score, float)
        assert len(r_type) == 1
        if aggregate_num > 1:
            cache_scores.append(score)
            cache_r_types.append(r_type[0][0])
            if (ind + 1) % aggregate_num == 0:
                scores.append(min(cache_scores))
                r_types.append((aggregate_reawrd_types(cache_r_types), None, None))
                cache_scores, cache_r_types = [], []
        else:
            scores.append(score)
            r_types += r_type
        fail_reasons += [fr[1].name if isinstance(fr, tuple) else str(fr) for fr in fail_reason]
        fail_indices += [fr[0] if isinstance(fr, tuple) else -1 for fr in fail_reason]
        num_inputs.append(num_input)
    return scores, r_types, fail_reasons, fail_indices, num_inputs


def eval_testcase(testcase_files, num_testcases=20, test_harness=True, aggregate_num=1, num_repeats=1, num_generators=5):
    datas = []
    added_files = []
    for file in testcase_files:
        if not os.path.exists(file):
            continue
        ds = datasets.load_dataset("parquet", data_files=[file])["train"]
        datas.append(ds)
        added_files.append(file)
    print(f"Added files: {added_files}")
    data = datasets.concatenate_datasets(datas)
    
    def check_input(item):
        return len(item["testcase"]) > 0 and all([tc != "" for tc in item["testcase"]])
    data = data.filter(check_input)

    print(f"====================== Evaluating {num_testcases} test cases ======================")
    pprint.pp(ray.available_resources())
    futures = []
    futures_map = {}
    for item in data:
        future = get_score_remote.remote(
            item,
            num_testcases=num_testcases,
            test_harness=test_harness,
            aggregate_num=aggregate_num,
            num_repeats=num_repeats,
            num_generators=num_generators,
        )
        futures.append(future)
        futures_map[future] = item
    
    results = []
    for future in tqdm(futures, total=len(futures)):
        item = futures_map[future]
        try:
            scores, r_types, fail_reasons, fail_indices, num_inputs = ray.get(future, timeout=2000)
        except GetTimeoutError:
            print("========== Timeout error, skipping this item ==========")
            scores = [-0.1] * len(item["testcase"])
            r_types = [(RewardType.FailGood, None, None)] * len(item["testcase"])
            fail_reasons = ["TimeoutError"] * len(item["testcase"])
            fail_indices = [0] * len(item["testcase"])
            num_inputs = [-1] * len(item["testcase"])
            ray.cancel(future, force=True, recursive=True)
        except (RayTaskError, WorkerCrashedError) as e:
            print(f"========== Ray error {e}, skipping this item ==========")
            scores = [-0.1] * len(item["testcase"])
            r_types = [(RewardType.FailGood, None, None)] * len(item["testcase"])
            fail_reasons = ["RayError"] * len(item["testcase"])
            fail_indices = [0] * len(item["testcase"])
            num_inputs = [-1] * len(item["testcase"])
            ray.cancel(future, force=True, recursive=True)
        
        item["score"] = scores
        item["reward_types"] = [rt[0].name for rt in r_types]
        item["fail_reasons"] = fail_reasons
        item["fail_indices"] = fail_indices
        item["num_inputs"] = [num[0] + num[1] if isinstance(num, tuple) else num for num in num_inputs]
        results.append(item)
    
    data = datasets.Dataset.from_list(results)
    def print_stats(ds):
        print(f"====================== Total # of data: {len(ds)}\t# of test cases {sum([len(x['testcase']) for x in ds])} ======================")
        reward_counts = []
        for i in range(len(ds[0]["reward_types"])):
            count_i = Counter([x["reward_types"][i] for x in ds])
            count_i = {k: v / sum(count_i.values()) * 100 for k, v in count_i.items()}
            reward_counts.append(count_i)
        print(f"Results over {len(reward_counts)} random seeds:")
        num_inputs = [i for l in ds["num_inputs"] for i in l if i >= 0]
        print(f"Num inputs: {np.mean(num_inputs):.1f} avg over {len(num_inputs)} test cases")
        good_input = [count.get("GoodInputFailGood", 0) + count.get("GoodInputPassGoodPassBad", 0) + count.get("PassGoodFailBad", 0) for count in reward_counts]
        print(f"Good Input: {np.mean(good_input):.1f} +/- {np.std(good_input):.1f}")
        fail_good = [count.get("FailGood", 0) + count.get("GoodInputFailGood", 0) for count in reward_counts]
        print(f"FailGood: {np.mean(fail_good):.1f} +/- {np.std(fail_good):.1f}")
        for key in ["PassGoodFailBad", "FailGood", "GoodInputFailGood", "PassGoodPassBad", "GoodInputPassGoodPassBad", "FormatError"]:
            result_i = [count.get(key, 0) for count in reward_counts]
            print(f"{key}: {np.mean(result_i):.1f} +/- {np.std(result_i):.1f}")
    print_stats(data)
    fail_reason_count = Counter([i for x in data["fail_reasons"] for i in x if i != "None"])
    total = sum(fail_reason_count.values())
    for key, count in fail_reason_count.items():
        print(f"{key}: {count / total:.4f}")


@ray.remote
def get_feedback_remote(item, num_testcases=20, test_harness=True, num_repeats=1, num_generators=5, num_cases_per_generator=4):
    testcase_judge.GLOBAL_TIMEOUT = 20  # timeout for evaluation
    num_passed = [0 for _ in range(len(item["reward_model"]["ground_truth"]))]
    num_tests = []
    for ind, tc in enumerate(item["testcase"]):
        num_correct, num_test = get_feedback(
            tc,
            item["reward_model"]["ground_truth"],
            num_testcases=num_testcases,
            test_harness=test_harness,
            num_repeats=num_repeats,
            timeout=20,
            seed=42,  # Use a fixed seed for reproducibility
            num_generators=num_generators,
            num_cases_per_generator=num_cases_per_generator,
        )
        assert len(num_correct) == len(item["reward_model"]["ground_truth"])
        for i, nc in enumerate(num_correct):
            num_passed[i] += nc
        num_tests.append(num_test)
    return num_passed, num_tests


def get_exec_feedback(testcase_files, outpath, num_testcases=20, test_harness=True, num_repeats=1, num_generators=5, num_cases_per_generator=4):
    datas = []
    added_files = []
    for file in testcase_files:
        if not os.path.exists(file):
            continue
        ds = datasets.load_dataset("parquet", data_files=[file])["train"]
        datas.append(ds)
        added_files.append(file)
    print(f"Added files: {added_files}")
    data = datasets.concatenate_datasets(datas)
    
    def check_input(item):
        return len(item["testcase"]) > 0 and all([tc != "" for tc in item["testcase"]])
    data = data.filter(check_input)

    print(f"====================== Getting feedback from {num_testcases} test cases ======================")
    pprint.pp(ray.available_resources())
    futures = []
    futures_map = {}
    for item in data:
        future = get_feedback_remote.remote(
            item,
            num_testcases=num_testcases,
            test_harness=test_harness,
            num_repeats=num_repeats,
            num_generators=num_generators,
            num_cases_per_generator=num_cases_per_generator,
        )
        futures.append(future)
        futures_map[future] = item
    
    results = []
    for future in tqdm(futures, total=len(futures)):
        item = futures_map[future]
        try:
            num_passed, num_tests = ray.get(future, timeout=7200)
        except GetTimeoutError:
            print("========== Timeout error, skipping this item ==========")
            num_passed = [0] * len(item["reward_model"]["ground_truth"])
            num_tests = [-1] * len(item["testcase"])
            ray.cancel(future, force=True, recursive=True)
        
        assert len(num_passed) == len(item["reward_model"]["ground_truth"])
        assert all([isinstance(n, int) for n in num_passed])
        assert len(num_tests) == len(item["testcase"])
        item["num_passed"] = num_passed
        item["num_tests"] = num_tests
        results.append(item)
    
    data = datasets.Dataset.from_list(results)
    print(f"====================== Total # of data: {len(data)}\t# of responses {sum([len(x['testcase']) for x in data])} ======================")
    data.to_parquet(outpath)


if __name__ == "__main__":
    args = parse_args()
    testcase_files = []
    for start_ind in range(500):
        filename = f"{args.input_dir}/{args.model.split('/')[-1]}_{args.prompt_file}_{start_ind}_{args.seed}.parquet"
        testcase_files.append(filename)
    test_harness= 'harness' in testcase_files[0]
    if args.mode == "eval_testcase":
        eval_testcase(testcase_files, num_testcases=args.num_testcases, test_harness=test_harness, aggregate_num=args.aggregate_num, num_repeats=args.num_repeats, num_generators=args.num_generators)
    elif args.mode == "get_feedback":
        outpath = f"{args.output_dir}/{args.model.split('/')[-1]}_{args.prompt_file}_{args.seed}_{args.num_testcases}_feedback.parquet"
        get_exec_feedback(testcase_files, outpath, num_testcases=args.num_testcases, test_harness=test_harness, num_repeats=args.num_repeats, num_generators=args.num_generators, num_cases_per_generator=args.num_cases_per_generator)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Supported modes are 'eval_testcase' and 'get_feedback'.")

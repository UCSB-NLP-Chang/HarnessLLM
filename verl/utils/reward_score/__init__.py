# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    configs=None,
    no_format_score=False,
):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    # NOTE: added
    elif "testcase" in data_source:
        from custom_verl import testcase_judge

        num_testcases = configs.get("num_testcases", 20) if configs is not None else 20
        wrong_test_penalty = (
            configs.get("wrong_test_penalty", -0.0) if configs is not None else -0.0
        )
        good_input_reward = (
            configs.get("good_input_reward", 0.1) if configs is not None else 0.1
        )
        test_harness = (
            configs.get("test_harness", True) if configs is not None else True
        )

        res = testcase_judge.compute_score(
            solution_str,
            ground_truth,
            num_testcases=num_testcases,
            wrong_test_penalty=wrong_test_penalty,
            test_harness=test_harness,
            good_input_reward=good_input_reward,
        )
    elif data_source.startswith("math") or data_source.startswith("AIME"):
        if configs is not None:
            math_func = configs.get("math_func", "default")
        else:
            math_func = "default"

        if math_func == "default":
            from . import math

            res = math.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        elif math_func == "qwen":
            from custom_verl.mathjudge import qwen_math

            #! For this function, we acceleate it by using multiprocessing and custom_naive.py
            res = qwen_math.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
            return res
        elif math_func == "mathverify":
            from custom_verl.mathjudge import math_verify

            res = math_verify.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        else:
            raise ValueError(f"Unknown math_func: {math_func}")

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        assert "testcase" in data_source
        reward_types = [i[0] for i in res[1]]
        fail_reasons = [i[1] if isinstance(i, tuple) else i for i in res[2]]
        num_tests = res[3][0] + res[3][1] if isinstance(res[3], tuple) else res[3]
        return float(res[0]), reward_types, fail_reasons, num_tests
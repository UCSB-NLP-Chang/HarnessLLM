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

import pprint
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from functools import partial

import numpy as np
import ray
import torch

from custom_verl.reward_utils import ExecutionResult, RewardType
from verl import DataProto
from verl.utils.reward_score import _default_compute_score


@ray.remote(num_cpus=1)
def _compute_score_remote(
    compute_score,
    data_source,
    solution_str,
    ground_truth,
    extra_info,
):
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


class CustomNaiveRewardManagerRay:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        configs=None,
        no_format_score=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        if configs is not None:
            # NOTE: we add configs that manipulates the compute_score function
            self.compute_score = partial(
                _default_compute_score, configs=configs, no_format_score=no_format_score
            )
            print("We're using the custom compute_score function")
        else:
            self.compute_score = compute_score or partial(
                _default_compute_score, no_format_score=no_format_score
            )
        self.TIMEOUT = 360

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        outcome_reward = torch.ones_like(reward_tensor[:, 0], dtype=torch.bfloat16)
        num_testcases = torch.ones_like(reward_tensor[:, 0], dtype=torch.bfloat16)
        batch_responses = [None] * len(data)
        reward_types = [None] * len(data)
        fail_reasons = [None] * len(data)
        max_response_length = data.batch["responses"].shape[1]

        futures, futures_map = [], {}
        already_print_data_sources = {}
        print("Available CPUs:")
        pprint.pp(ray.available_resources())
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            batch_responses[i] = sequences_str
            solution_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"][
                "ground_truth"
            ]
            data_source = data_item.non_tensor_batch["data_source"]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
            
            # Submit the computation to Ray
            future = _compute_score_remote.remote(
                self.compute_score,
                data_source,
                solution_str,
                ground_truth,
                extra_info,
            )
            futures.append(future)
            futures_map[future] = (i, valid_response_length)
        
        # Now gather results
        print("=" * 100 + "\nStarting to collect results...\n" + "=" * 99)
        ready, not_ready = ray.wait(futures, timeout=self.TIMEOUT, num_returns=len(futures))
        for future in ready:
            i, valid_response_length = futures_map[future]
            score_res = ray.get(future)
            reward_tensor[i, valid_response_length - 1] = score_res[0]
            outcome_reward[i] = score_res[0]
            reward_types[i] = score_res[1]
            fail_reasons[i] = score_res[2]
            num_testcases[i] = score_res[3]
        # If there are any futures that are not ready, we handle them
        if not_ready:
            print("=" * 100 + f"\nTimeoutError: {len(not_ready)} futures took too long to complete.\n" + "=" * 99)
            for future in not_ready:
                i, valid_response_length = futures_map[future]
                # If the future is not done, we just set a default value
                reward_tensor[i, valid_response_length - 1] = 0.0
                outcome_reward[i] = 0.0
                reward_types[i] = [RewardType.FailGood]
                fail_reasons[i] = [ExecutionResult.TimeoutError]
                # Cancel the future if it's not done
                ray.cancel(future, force=True, recursive=True)

        reward_tensor = {
            "token_level_scores": reward_tensor,
            "outcome_reward": outcome_reward,
            "num_testcases": num_testcases,
            "batch_responses": np.array(batch_responses, dtype=object),
            "reward_types": np.array(reward_types, dtype=object),
            "fail_reasons": np.array(fail_reasons, dtype=object),
        }
        if "data_source" in data.non_tensor_batch:
            reward_tensor["data_source"] = data.non_tensor_batch["data_source"]
        if "uid" in data.non_tensor_batch:
            reward_tensor["uid"] = data.non_tensor_batch["uid"]

        return DataProto.from_single_dict(reward_tensor)
import json
import os
import random
from abc import ABC
from collections import Counter, defaultdict

import numpy as np

from verl.protocol import DataProto
from verl.utils.tracking import Tracking


class Callback(ABC):
    def __init__(self, logger: Tracking, **kwargs):
        super().__init__()
        self.logger = logger

    def on_generate_sequences(self, global_step, batch: DataProto, split: str):
        pass

    def on_val_end(self, global_step, batch: DataProto, split: str):
        pass


class RewardTypeCallback(Callback):
    def on_generate_sequences(self, global_step, batch, split):
        keyname = "reward_types"
        if keyname not in batch.non_tensor_batch:
            return

        reward_type_count = batch.non_tensor_batch[keyname]
        if isinstance(reward_type_count[0], list) or isinstance(
            reward_type_count[0], np.ndarray
        ):
            reward_type_count = [
                item for sublist in reward_type_count for item in sublist
            ]
        type_counts = Counter(reward_type_count)
        total_count = sum(type_counts.values())
        json_counts = {
            f"{split}/reward_type/{reward.name}": count / total_count
            for reward, count in type_counts.items()
        }
        # Aggregate by group
        for aggregate_key in ["data_source"]:
            if aggregate_key not in batch.non_tensor_batch:
                continue
            group_type_counts = defaultdict(Counter)
            group_uid_score_counts = defaultdict(lambda: defaultdict(set))
            for i, reward_type in enumerate(batch.non_tensor_batch[keyname]):
                group_key = batch.non_tensor_batch[aggregate_key][i]
                if not (
                    isinstance(reward_type, list) or isinstance(reward_type, np.ndarray)
                ):
                    reward_type = [reward_type]
                group_type_counts[group_key].update(reward_type)
                uid = batch.non_tensor_batch["uid"][i]
                score_i = batch.batch["outcome_reward"][i].item()
                group_uid_score_counts[group_key][uid].add(score_i)
            for group_key, type_counts in group_type_counts.items():
                total_count = sum(type_counts.values())
                json_counts.update(
                    {
                        f"{split}/reward_type/{aggregate_key}/{group_key}/{reward.name}": count
                        / total_count
                        for reward, count in type_counts.items()
                    }
                )
            # check how many unique problems have only one type of score
            for group_key, uid_type_counts in group_uid_score_counts.items():
                unique_problems = len(uid_type_counts)
                single_scores = [list(val)[0] for val in uid_type_counts.values() if len(val) == 1]
                single_score_count = Counter(single_scores)
                json_counts.update(
                    {
                        f"{split}/reward_type/{aggregate_key}/{group_key}/single_reward/{score}": count
                        / unique_problems
                        for score, count in single_score_count.items()
                    }
                )
        self.logger.log(data=json_counts, step=global_step)


class TestCaseTypeCallback(Callback):
    def __init__(self, logger, keyword_file: str = "keywords.json"):
        super().__init__(logger=logger)
        self.keywords = json.load(open(keyword_file, "r"))

    def on_generate_sequences(self, global_step: int, batch: DataProto, split: str):
        import re

        required_keys = [
            "batch_responses",  # responses
        ]
        for keyname in required_keys:
            if keyname not in batch.non_tensor_batch:
                return

        res_list = defaultdict(list)
        response_strs = batch.non_tensor_batch["batch_responses"]
        brute_patterns = [r"\bbrute\b", r"^\s*def\s+\w*correct\w*\s*\("]
        complex_input_rewards = []
        fail_reasons, complex_input_fail_reasons = [], []

        for sample_index, response in enumerate(response_strs):
            res = {
                "brute_force": False,
                "property": False,
                "complex_input": False,
            }
            matches = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
            if not matches:
                for k, v in res.items():
                    res_list[k].append(v)
                continue
            response = matches[-1]
            for type_key, kwords in self.keywords.items():
                if any([kw in response.lower() for kw in kwords]):
                    res[type_key] = True
            if not res["brute_force"]:
                for pattern in brute_patterns:
                    if re.search(pattern, response.lower()):
                        res["brute_force"] = True
                        break
            if not res["property"]:
                lines = response.split("\n")
                lines = [l for l in lines if "assert" in l]
                # remove comments
                lines = [l.split("#")[0].split('"')[0] for l in lines]
                lines = [l for l in lines if any([kw in l for kw in ["!=", ">", "<", "check", "valid"]])]
                if len(lines) > 0:
                    res["property"] = True
            if res["complex_input"]:
                complex_input_rewards.extend(
                    batch.non_tensor_batch["reward_types"][sample_index]
                )
                complex_input_fail_reasons.extend(
                    batch.non_tensor_batch["fail_reasons"][sample_index]
                )
            fail_reasons.extend(
                batch.non_tensor_batch["fail_reasons"][sample_index]
            )
            for k, v in res.items():
                res_list[k].append(v)

        count_res = {
            f"{split}/testcase_type/{key}": sum(count) / len(count)
            for key, count in res_list.items()
        }
        count_res[f"{split}/testcase_type/num_testcase"] = batch.batch["num_testcases"].mean().item()
        type_counts = Counter(complex_input_rewards)
        total_count = sum(type_counts.values())
        count_res.update(
            {
                f"{split}/reward_type/complex_input/{reward.name}": count / total_count
                for reward, count in type_counts.items()
            }
        )
        fail_reasons = [i for i in fail_reasons if i is not None]
        fail_reason_counts = Counter(fail_reasons)
        total_count = sum(fail_reason_counts.values())
        count_res.update(
            {
                f"{split}/fail_reason/{reason.name}": count / total_count
                for reason, count in fail_reason_counts.items()
            }
        )
        complex_input_fail_reasons = [i for i in complex_input_fail_reasons if i is not None]
        complex_fail_reason_counts = Counter(complex_input_fail_reasons)
        total_count = sum(complex_fail_reason_counts.values())
        count_res.update(
            {
                f"{split}/complex_fail_reason/{reason.name}": count / total_count
                for reason, count in complex_fail_reason_counts.items()
            }
        )
        # Aggregate by group
        for aggregate_key in ["data_source"]:
            if aggregate_key not in batch.non_tensor_batch:
                continue
            group_res_list = defaultdict(lambda: {k: [] for k in res_list.keys()})
            for i, group_key in enumerate(batch.non_tensor_batch[aggregate_key]):
                for k, v in res_list.items():
                    group_res_list[group_key][k].append(v[i])
            for group_key, res in group_res_list.items():
                count_res.update(
                    {
                        f"{split}/testcase_type/{aggregate_key}/{group_key}/{key}": sum(
                            count
                        )
                        / len(count)
                        for key, count in res.items()
                    }
                )
        self.logger.log(data=count_res, step=global_step)
        return count_res


class ValSaveCallback(Callback):
    def __init__(self, logger, output_dir, **kwargs):
        super().__init__(logger, **kwargs)
        self.output_dir = output_dir

    def on_val_end(self, global_step, batch, split):
        os.makedirs(self.output_dir, exist_ok=True)
        all_reuslts = []
        for i in range(len(batch.non_tensor_batch["batch_responses"])):
            all_reuslts.append(
                {
                    "uid": batch.non_tensor_batch["uid"][i],
                    "input": batch.non_tensor_batch["batch_inputs"][i],
                    "response": batch.non_tensor_batch["batch_responses"][i],
                    "score": batch.non_tensor_batch["scores"][i],
                    "reward_types": [
                        x.name for x in batch.non_tensor_batch["reward_types"][i]
                    ],
                }
            )

        with open(
            os.path.join(self.output_dir, f"{split}_step@{global_step}.jsonl"), "w"
        ) as f:
            for line in all_reuslts:
                f.write(json.dumps(line) + "\n")


class CallbackManager(Callback):
    def __init__(self, logger: Tracking, rootdir=None, **kwargs):
        super().__init__(logger=logger, **kwargs)

        # TODO (jiabao): we may add more callbacks here
        self.callbacks = []
        self.callbacks.append(RewardTypeCallback(self.logger))
        self.callbacks.append(TestCaseTypeCallback(self.logger))
        if rootdir is not None:
            self.callbacks.append(
                ValSaveCallback(self.logger, os.path.join(rootdir, "evallog"))
            )

    def on_generate_sequences(self, step, batch: DataProto, split: str):
        for callback in self.callbacks:
            callback.on_generate_sequences(step, batch, split)

    def on_val_end(self, step, batch: DataProto, split: str):
        for callback in self.callbacks:
            callback.on_val_end(step, batch, split)

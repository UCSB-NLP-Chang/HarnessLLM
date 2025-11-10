import copy
import json
import os
from typing import Any, Dict, List

import datasets
from omegaconf import ListConfig

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from custom_verl.prompt_utils import (
    find_format_map_names,
    load_constants,
    replace_prompt,
)


class HFRLHFDataset(RLHFDataset):
    
    def process_sample(
        self, prompt_with_chat_template: str, row_dict: Dict[str, Any], chat
    ) -> Dict[str, Any]:
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item]
        row_dict = copy.deepcopy(row_dict)
        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        return self.process_sample(prompt_with_chat_template, row_dict, chat)


class DynamicPromptRLHFDataset(HFRLHFDataset):
    # This class dynamiclly replaces the prompt with the template in prompt_file.
    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor,
        prompt_file=None,
    ):
        self.MESSAGE_TEMPLATE = load_constants(
            prompt_file
        )  # We apply differnt prompt dynamically, to avoid waste of hard drive space
        if isinstance(self.MESSAGE_TEMPLATE, dict):
            template = list(self.MESSAGE_TEMPLATE.values())[0]
        else:
            template = self.MESSAGE_TEMPLATE
        required_names = find_format_map_names(
            "\n".join([x["content"] for x in template])
        )
        self.required_names = [x for x in required_names if x != ""]

        super().__init__(
            data_files,
            tokenizer,
            config,
            processor,
        )

    def prompt_fn(self, sample):
        sample = copy.deepcopy(sample)
        replace_dict = {key: sample.pop(key) for key in self.required_names}
        res = replace_prompt(self.MESSAGE_TEMPLATE, replace_dict)
        return res
    
    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            if os.path.exists(data_file):
                # read parquet files and cache
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            else:
                dataframe = datasets.load_dataset(data_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe.filter(
                lambda doc: len(self.tokenizer.apply_chat_template(self.prompt_fn(doc), add_generation_prompt=True)
                               ) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

            print(f'filter dataset len: {len(self.dataframe)}')

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]

        chat = self.prompt_fn(row_dict)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        return self.process_sample(prompt_with_chat_template, row_dict, chat)


class DynamicTestCaseCodeRLHFDataset(DynamicPromptRLHFDataset):
    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor,
        prompt_file=None,
    ):
        super().__init__(
            data_files,
            tokenizer,
            config,
            processor,
            prompt_file,
        )

    def prompt_fn(self, sample):
        if isinstance(sample["reward_model"]["ground_truth"], str):
            sample["reward_model"]["ground_truth"] = json.loads(
                sample["reward_model"]["ground_truth"]
            )

        bad_code = sample["reward_model"]["ground_truth"]["bad_code"]
        sample["testing_code"] = bad_code["code"] if isinstance(bad_code, dict) else bad_code
        replace_dict = {key: sample.pop(key) for key in self.required_names}
        res = replace_prompt(self.MESSAGE_TEMPLATE[sample["reward_model"]["ground_truth"].get("code_type", "input/output")], replace_dict)
        return res
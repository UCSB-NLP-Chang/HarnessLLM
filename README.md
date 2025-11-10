# HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning

This is the implementation for the paper [HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning](https://arxiv.org/pdf/2511.01104). We propose a framework that trains LLMs to generate automatic test harnesses.

## Quick Links
- [Collection on Huggingface](https://huggingface.co/collections/Shiyu-Lab/harnessllm)
- [RL Training Data on Huggingface](https://huggingface.co/datasets/Shiyu-Lab/Testcase_RL_Data)
- [Model Checkpoints on Huggingface](https://huggingface.co/Shiyu-Lab/HarnessLLM_RL_Qwen3_4B)

## Installation

```bash
conda create -n harness python=3.10 -y
conda activate harness
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
conda install conda-forge::python-prctl
```

## Evaluation

First, generate test cases with a model on a specific dataset:
```
bash bashes/generate.sh {model} {data} {prompt} {port}
```
where `prompt` can be `harness` or `inputoutput`. For example, to run our model on LCB Seen version, run:
```
bash bashes/generate.sh Shiyu-Lab/HarnessLLM_RL_Qwen3_4B Shiyu-Lab/Testcase_LCB_Seen harness 30000
```
Next, evaluate the generated test cases with:
```
python -m scripts.eval {model} {prompt} --data_path {data}
```
For example, to evaluate the above-generated test cases, run:
```
python -m scripts.eval Shiyu-Lab/HarnessLLM_RL_Qwen3_4B harness --data_path Shiyu-Lab/Testcase_LCB_Seen
```
You can also run `bash bashes/generate_and_eval_testcase.sh` to evaluate on all datasets.

## Training

### SFT Training
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT training.
You can download our training data from [Huggingface](https://huggingface.co/datasets/Shiyu-Lab/HarnessLLM_SFT_Data). The config file we use for SFT training is `scripts/qwen3_sft.yaml`. You can follow the instructions on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples#supervised-fine-tuning-on-multiple-nodes-1) to launch training.

### RL Training
Use the script `examples/grpo_trainer/run_harness.sh` to train our model.

## Citation
```
@misc{liu2025harnessllmautomatictestingharness,
      title={HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning}, 
      author={Yujian Liu and Jiabao Ji and Yang Zhang and Wenbo Guo and Tommi Jaakkola and Shiyu Chang},
      year={2025},
      eprint={2511.01104},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2511.01104}, 
}
```

## Acknowledgement
Our implementation is based on [Verl](https://github.com/volcengine/verl).

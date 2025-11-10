MODEL=Shiyu-Lab/HarnessLLM_RL_Qwen3_4B
PROMPT=harness
PORT=30000

# Generation
for data in Shiyu-Lab/Testcase_MBPPHard Shiyu-Lab/Testcase_LCB_Seen Shiyu-Lab/Testcase_LCB_Unseen Shiyu-Lab/Testcase_CF_Seen Shiyu-Lab/Testcase_CF_Unseen
do
    bash bashes/generate.sh $MODEL $data $PROMPT $PORT
done

# Evaluation
for data in Shiyu-Lab/Testcase_MBPPHard Shiyu-Lab/Testcase_LCB_Seen Shiyu-Lab/Testcase_LCB_Unseen Shiyu-Lab/Testcase_CF_Seen Shiyu-Lab/Testcase_CF_Unseen
do
    echo "Model: $MODEL"
    echo "Data: $data"
    echo "Prompt: $PROMPT"
    python -m scripts.eval $MODEL $PROMPT --data_path $data
done
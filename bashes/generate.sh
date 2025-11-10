model_name=$1
data=$2
prompt=$3
port=$4

python -m sglang.launch_server --model-path $model_name --tp 2 --port $port &
LM_PID=$!

sleep 60

python -m scripts.generate $model_name $prompt $data --port $port --num_sample 8 --num_process 8

kill $LM_PID
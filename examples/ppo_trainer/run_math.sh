set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

N_GPUS=8
TP=1
MODEL_DIR=/data/yujian_liu/math/ckpts/Qwen-7B_ppo
DATA_DIR=/data/yujian_liu/math/data/verl_train
BATCH_SIZE=8
FW_BS=$((BATCH_SIZE * 2))

python3 -m verl.trainer.main_ppo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=3000 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$FW_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-Math-7B \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=$BATCH_SIZE \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$FW_BS \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    +trainer.val_before_train=False \
    trainer.default_local_dir=$MODEL_DIR \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name='qwen2.5_7b_ppo' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 $@

# actor_rollout_ref.ref.fsdp_config.param_offload=True \
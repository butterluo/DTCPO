#!/bin/bash
set -e -x
FILE="$(pwd)/my_reward_function.py"
rm -rf $FILE
cat <<EOF > "$FILE"
def my_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    print(f"Congratulations!!! You have called my_reward_function successfully!!!")
    return 0.1
EOF


OUTPUT_FILE="$(pwd)/output_custom_reward.txt"
FUNCTION_NAME="my_reward_function"
rm -rf $OUTPUT_FILE

huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir $HOME/models/Qwen/Qwen2.5-0.5B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$HOME/models/Qwen/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$HOME/models/Qwen/Qwen2.5-0.5B \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$FILE\
    custom_reward_function.name=$FUNCTION_NAME\
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='qwen_e2e_ci_custom_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$HOME/ckpt/ \
    trainer.total_training_steps=2 | tee $OUTPUT_FILE;

python3 tests/e2e/check_custom_rwd_fn.py --output_file=$OUTPUT_FILE
rm -rf $FILE
rm -rf $OUTPUT_FILE
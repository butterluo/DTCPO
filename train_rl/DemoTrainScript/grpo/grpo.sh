## nohup ./grpo.sh > L0504.log 2>&1 &
set -x

EVRL_PATH="/home/azureuser/cloudfiles/code/SRC/RL/verl"
RSN_PATH="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/anaconda/envs/azureml_py38/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"
export DISABLE_MLFLOW_INTEGRATION='TRUE'
export TF_ENABLE_ONEDNN_OPTS="0"
export PYTHONPATH=$EVRL_PATH:$RSN_PATH:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HYDRA_FULL_ERROR=1

export VLLM_ATTENTION_BACKEND=XFORMERS


TRN_DS=$RSN_PATH/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K_vrl/train.parquet
TST_DS=$RSN_PATH/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K_vrl/test.parquet
ACT_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
RWD_FUNC_PATH=$EVRL_PATH/BT/cuz/rwdfunc/medVryEsyGrpo.py

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRN_DS \
    data.val_files=$TST_DS \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=10128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACT_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_batched_tokens=102400 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=FreedomIntelligence/medical_o1_verifier_3B \
    reward_model.max_length=10240 \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$RWD_FUNC_PATH \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    +trainer.tensorboard_dir="${PWD}/tblog" \
    trainer.project_name='092rl16' \
    trainer.experiment_name='0504' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=62 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=10
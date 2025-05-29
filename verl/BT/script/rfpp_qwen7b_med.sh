## nohup ./rfpp_qwen7b_med.sh > Lrfpp_qwen7b_med0411.log 2>&1 &
set -x

EVRL_PATH="/home/azureuser/cloudfiles/code/SRC/RL/verl"
RSN_PATH="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove"

export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES="0,1"
export LD_LIBRARY_PATH=/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"
export DISABLE_MLFLOW_INTEGRATION='TRUE'
export TF_ENABLE_ONEDNN_OPTS="0"
export PYTHONPATH=$EVRL_PATH:$PYTHONPATH

export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR="rfppqwn_tblog"

export VLLM_ATTENTION_BACKEND=XFORMERS


TRN_DS=$RSN_PATH/domain/Med/data/tmpds/med-o1_VScr_easy4RL1k_vrl/train.parquet
TST_DS=$RSN_PATH/domain/Med/data/tmpds/med-o1_VScr_easy4RL1k_vrl/test.parquet
ACT_PATH=$RSN_PATH/domain/Med/1D5B/chkp_dir/qwn7bVryesyGrpo_0407/checkpoint-500
# ACT_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# ACT_PATH=Qwen/Qwen2.5-0.5B
RWD_FUNC_PATH=$EVRL_PATH/BT/cuz/rwdfunc/medVryEsyGrpo.py

# val_batch_size不用了,做eval是整个train_batch_size给vllm/sglang让它自己去切分
# use_liger=True目前只能用在sft
# rollout.max_num_batched_tokens>=max_prompt_length+max_response_length
#@#TODO 暂不支持在有reward model/verifier的情况下做validate,所以val_before_train=False和test_freq=-1。但特么又非得构建validate所需要的dataset,所以还是要val_files
#@# debug这个shell中的py,参考 https://blog.csdn.net/m0_52394190/article/details/136913701
#@# ray_trainer.py中‘real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n’
#@# 要保证ppo_mini_batch_size = (ppo_mini_batch_size*rollout_n) // (world_size // ulysses_sequence_parallel_size) 大于等于 ppo_micro_batch_size_per_gpu，否则由于“radient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu会导致gradient_accumulation为0进而产生nan
# python3 -m verl.trainer.main_ppo \
# raise-exception
# python -m debugpy --listen localhost:15678 --wait-for-client \
#    /home/azureuser/cloudfiles/code/SRC/RL/verl/verl/trainer/main_ppo.py \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$TRN_DS \
    data.val_files=$TST_DS \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACT_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12352 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.project_name='medrsn' \
    trainer.experiment_name='qwn7RfppVryEsyGrpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=5
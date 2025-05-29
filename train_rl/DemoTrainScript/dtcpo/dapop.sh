#!/usr/bin/env bash
set -euxo pipefail
## nohup ./dapop.sh > L0516.log 2>&1 & 

project_name='0122'
exp_name='0514'

EVRL_PATH="/home/azureuser/cloudfiles/code/SRC/RL/verl"
RSN_PATH="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove"

export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES="0,1"
# LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}
# export LD_LIBRARY_PATH="/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH"
# export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"
PYTHONPATH=${PYTHONPATH:-""}
export PYTHONPATH=$EVRL_PATH:$PYTHONPATH
export HYDRA_FULL_ERROR=1

TENSORBOARD_DIR="${PWD}/tblog"
SUBMITID="Med${project_name}${exp_name}$(date +'%y%m%d%H%M')"
echo "==SUBMITID==>> ${SUBMITID}"

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_final_reward 
max_num_gen_batches=10

NUM_GPUS=4  # n
train_traj_micro_bsz_per_gpu=8 # b
train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
micro_mini_bsz_factor=2 # f
train_traj_mini_bsz=$((train_traj_micro_bsz * micro_mini_bsz_factor)) # b * n * f
n_resp_per_prompt=8  #rollout.n # g
train_prompt_mini_bsz=$((train_traj_mini_bsz / n_resp_per_prompt)) # b * n * f / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # b * n * f * 1 / g
gen_prompt_bsz=$((train_prompt_bsz * 1))

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:2266"}
RAY_ADDRESS="http://localhost:8265"
WORKING_DIR=${EVRL_PATH}
# RUNTIME_ENV="${RSN_PATH}/domain/Med/comm/rayrt.yaml"
# NNODES=${NNODES:-16}
# Paths
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
RWD_FUNC_PATH=$EVRL_PATH/BT/cuz/rwdfunc/medRwdMrge.py
CKPTS_DIR="${PWD}/ckpt/${exp_name}"
TRAIN_FILE=$RSN_PATH/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K_vrl/train.parquet
TEST_FILE=$RSN_PATH/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K_vrl/test.parquet

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))
actor_ppo_max_token_len=$((2 * max_response_length))
infer_ppo_max_token_len=$((2 * max_response_length))
max_num_batched_tokens=$((infer_ppo_max_token_len * 5))
rwdmdl_max_token_len=$((512 + max_response_length))
vllm_tp=1

python3 -m recipe.dapo.src.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    +data.stdZroSmpThrshld=4 \
    +data.stdZroHrdSmpEsyThrshld=1.14 \
    +data.stdZroHrdSmpRetrainThrshld=2 \
    +data.stdZroSmpReszFactr=0.6 \
    +data.stdZroHrdSmpThrshld=0.3 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.seedChangeEpoch=False \
    +actor_rollout_ref.rollout.seedChangeEpochStepTrshldFactor=0.2 \
    +actor_rollout_ref.rollout.stdZroSmpIncrTmperTrshld=0.5 \
    +actor_rollout_ref.rollout.stdZroSmpIncrTmper=0.2 \
    +actor_rollout_ref.rollout.stdZroSmpIncrTmperMx=1.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.enable=True \
    reward_model.model.path=FreedomIntelligence/medical_o1_verifier_3B \
    reward_model.max_length=${rwdmdl_max_token_len} \
    reward_model.micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    reward_model.model.fsdp_config.param_offload=False \
    reward_model.model.use_remove_padding=False  \
    custom_reward_function.path=$RWD_FUNC_PATH \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=btdapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','tensorboard'] \
    +trainer.tensorboard_dir="${TENSORBOARD_DIR}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    +trainer.saveEpoch=3  \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
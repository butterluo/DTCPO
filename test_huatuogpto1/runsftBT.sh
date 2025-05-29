export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH 
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="~/cloudfiles/code/Cache/HF/"
export ACCELERATE_LOG_LEVEL=info
export DISABLE_MLFLOW_INTEGRATION='TRUE'
# export WANDB_DISABLED="true"
export TF_ENABLE_ONEDNN_OPTS=0
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
# export OMP_NUM_THREADS=1
export TORCHDYNAMO_VERBOSE=0


accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 2  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1BT.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --data_path /home/azureuser/cloudfiles/code/reasoningimprove/domain/Med/data/tmpds/med-R1-DSTL2k1w_rsnmth6k1k \
    --experiment_name sft0327
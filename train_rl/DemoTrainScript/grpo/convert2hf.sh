EVRL_PATH="/home/azureuser/cloudfiles/code/SRC/RL/verl"
RSN_PATH="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove"
CKPBASE="$RSN_PATH/domain/Med/1D5B/Demo/grpo/checkpoints/092rl16/0504"
CKPBASEHF="${CKPBASE}HF"

export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES="0,1"
export LD_LIBRARY_PATH=/anaconda/envs/azureml_py38/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"
export DISABLE_MLFLOW_INTEGRATION='TRUE'
export TF_ENABLE_ONEDNN_OPTS="0"
export PYTHONPATH=$EVRL_PATH:$RSN_PATH:$PYTHONPATH

export HYDRA_FULL_ERROR=1

export VLLM_ATTENTION_BACKEND=XFORMERS
#
for ckp in "global_step_620" "global_step_310"
do
  ckppath="${CKPBASE}/${ckp}/actor"
  ckppathHf="${CKPBASEHF}/${ckp}"
  echo "Converting ${ckppath}"
  python $EVRL_PATH/scripts/model_merger.py \
      --backend fsdp \
      --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --local_dir ${ckppath} \
      --target_dir ${ckppathHf}
  
  
  echo "Converted to ${ckppathHf}"
done


## nohup ./eval.sh > Leval0425.log 2>&1 &

echo "----0----"

PRJBASE="/home/azureuser/cloudfiles/code/SRC/Domain/Med/HuatuoGPT-o1"
CKPBASE="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove/domain/Med/1D5B/Expm/012Vq1kDapo2DstlQwn2/ckpt/0514HF"

export PYTHONUNBUFFERED=1
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}
export CUDA_VISIBLE_DEVICES="0"
export LD_LIBRARY_PATH="/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH"
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
export HF_HOME="/home/azureuser/cloudfiles/code/Cache/HF/"
PYTHONPATH=${PYTHONPATH:-""}
export PYTHONPATH=$PRJBASE:$PYTHONPATH

echo "----1----"

log_num=0
port=28${log_num}35
# global_step_620" "global_step_569" "global_step_363" "global_step_216"
for i in "global_step_301"
do
  model_name="${CKPBASE}/${i}"

  rm -rf ~/.cache/flashinfer/*
  echo $model_name
  python -m sglang.launch_server --model-path $model_name --host 0.0.0.0 --port $port > sglang${log_num}.log 2>&1 &

  sleep 150s

  python $PRJBASE/bt/eval/eval.py --model_name $model_name  --eval_file $PRJBASE/evaluation/data/eval_data.json --port $port 

  bash $PRJBASE/evaluation/kill_sglang_server.sh

  sleep 1m
done

# 
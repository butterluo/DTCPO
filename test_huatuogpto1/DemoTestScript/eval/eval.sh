PRJBASE="/home/azureuser/cloudfiles/code/SRC/Domain/Med/HuatuoGPT-o1"
CKPBASE="/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove/domain/Med/1D5B/Expm/001R1DstlLen1wSft/chkp/041501"

export PYTHONUNBUFFERED=1

log_num=0
port=28${log_num}35
#"checkpoint-43" "checkpoint-86" "checkpoint-129" "checkpoint-172" "checkpoint-215"
for i in "checkpoint-172" "checkpoint-215" 
do
  model_name="$CKPBASE/$i"

  rm -rf ~/.cache/flashinfer/*
  echo $model_name
  python -m sglang.launch_server --model-path $model_name --host 0.0.0.0 --port $port > sglang${log_num}.log 2>&1 &

  sleep 2m

  python $PRJBASE/bt/eval/eval.py --model_name $model_name  --eval_file $PRJBASE/evaluation/data/eval_data.json --port $port 

  bash $PRJBASE/evaluation/kill_sglang_server.sh

  sleep 1m
done
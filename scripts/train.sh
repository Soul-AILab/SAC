#!/bin/bash

# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="your_wandb_key"  # Change this to your actual wandb key
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir="/path/to/SAC"  # Change this to your actual root directory

# Set the run name
run_name="sac_16khz_train_37_5hz"

# Set default parameters
log_dir="/path/to/exp/$run_name"
nnodes=1
nproc_per_node=8
num_workers=16
config="configs/sac_16k_37_5.yaml"
train_engine="deepspeed"
deepspeed_config="configs/ds_stage2.json"
resume_step=0
debug=false
project='codec_sac'

port=10086

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --nnodes)
      nnodes="$2"
      shift 2
      ;;
    --gpus|--nproc_per_node)
      nproc_per_node="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done


cd "$root_dir" || exit
source utils/parse_options.sh

# Check if log_dir is already an absolute path
if [[ "$log_dir" != /* ]]; then
    log_dir="$root_dir/$log_dir"
fi

# Check if log directory exists
if [ $resume_step -eq 0 ]; then
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
        echo "Log directory created: $log_dir"
    elif [ "$debug" = false ]; then
        echo "Error: Log directory '$log_dir' already exists. Please remove or choose a different directory."
        exit 1
    fi
fi

# Write command to run.bash
tag="$(date +'%Y%m%d_%H%M%S')"

cat <<EOF > "$log_dir/${tag}_run.sh"
#!/bin/bash

# # Change directory to the root directory
cd "$root_dir" || exit

torchrun --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} --master_port=${port} \\
        -m bins.train \\
        --config ${config} \\
        --log_dir ${log_dir} \\
        --train_engine ${train_engine} \\
        --deepspeed_config ${deepspeed_config} \\
        --resume_step ${resume_step} \\
        --date ${tag} \\
        --project ${project} \\
        --enable_wandb \\
        --wandb_runs_name ${run_name} \\
EOF

chmod +x "$log_dir/${tag}_run.sh"
echo "run bash is saved to $log_dir/${tag}_run.sh"

echo "execute $log_dir/${tag}_run.sh"

bash "$log_dir/${tag}_run.sh"

# Example command:
# bash scripts/train.sh
# bash scripts/train.sh --nnodes 1 --gpus 8
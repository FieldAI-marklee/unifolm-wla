export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# Fine-tune a released UnifoLM-WLA-Base checkpoint (frozen VLM, single 24GB GPU).
#
# === Please modify the following paths according to your environment ===
# Each can also be overridden from the environment, e.g.
#   config_yaml=... data_config_path=... run_id=... ACCELERATE=... WANDB_MODE=offline bash <this script>
base_model_dir=${base_model_dir:-/path/to/UnifoLM-WLA-1.0-Base}
config_yaml=${config_yaml:-./unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml}
data_config_path=${data_config_path:-./unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-finetune_wla_base_frozen_vlm}
ACCELERATE=${ACCELERATE:-.venv/bin/accelerate}
# deepspeed_zero2_offload.yaml keeps fp32 master weights + Adam states in CPU RAM (needed on a single <=32 GB GPU)
ds_launch_config=${ds_launch_config:-unifolm_wla/config/deepseeds/deepspeed_zero2.yaml}
# === End of environment variable configuration ===
###########################################################################################
# offline: runs are written to <output_dir>/wandb (upload later with `wandb sync`); online needs `wandb login`
export WANDB_MODE=${WANDB_MODE:-offline}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-1}

${ACCELERATE} launch \
  --config_file ${ds_launch_config} \
  --num_processes ${num_processes} \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml ${config_yaml} \
  --framework.qwenvl.base_vlm ${base_model_dir}/tokenizer \
  --trainer.pretrained_checkpoint ${base_model_dir}/checkpoints/model.safetensors \
  --datasets.vla_data.data_config_path ${data_config_path} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  "$@" 2>&1 | tee -a ${output_dir}/train.log
exit ${PIPESTATUS[0]}

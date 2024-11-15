#!/bin/sh
#SBATCH --partition=4090
#SBATCH --output=slurm_out/slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=200G

{
 REQUIRED_GPUS=5
 

get_free_gpus() {
    local gpus=()
    local nvidia_output=$(nvidia-smi | grep ' C ' | awk '{print $2}')
    for gpu_id in $nvidia_output; do
        gpus+=("$gpu_id")
    done
    echo "${gpus[@]}"
}


while true; do
  # 获取所有正在使用的 GPU ID
  used_gpus=$(get_free_gpus)

  # 获取所有 GPU 总数
  total_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

  # 获取空闲 GPU ID（总 GPU 减去使用中的 GPU）
  free_gpus=()
  for i in $(seq 0 $(($total_gpus - 1))); do
    if [[ ! " ${used_gpus[@]} " =~ " $i " ]]; then
      free_gpus+=($i)
    fi
  done

  # 检查是否有指定数量的空闲 GPU
  if [ ${#free_gpus[@]} -ge $REQUIRED_GPUS ]; then
    # 将前 N 个空闲 GPU ID 赋值给 CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(echo "${free_gpus[@]:0:$REQUIRED_GPUS}" | tr ' ' ',')
    echo "找到 $REQUIRED_GPUS 个空闲 GPU，ID 为：$CUDA_VISIBLE_DEVICES"
    break
  else
    echo "当前空闲 GPU 数量不足 $REQUIRED_GPUS 个，继续等待..."
  fi
  # 等待5秒后再次检查
  sleep 5
done


# 随机端口生成函数
random_port() {
  echo $((RANDOM % 55535 + 10000))  # 范围：10000-65535
}

# 函数：检查端口是否被占用
is_port_in_use() {
  local port=$1
  if lsof -i:"$port" >/dev/null 2>&1; then
    return 0  # 端口被占用
  else
    return 1  # 端口未被占用
  fi
}

# 找到未占用的端口
while true; do
  MASTER_PORT=$(random_port)
  if ! is_port_in_use "$MASTER_PORT"; then
    echo "使用端口：$MASTER_PORT"
    break
  fi
done

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$REQUIRED_GPUS --master_port=$MASTER_PORT src/distilling/main.py
#CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 --master_port=$MASTER_PORT src/distilling/main.py
#CUDA_VISIBLE_DEVICES=0,1,2,3, torchrun --nproc_per_node=4 --master_port=$MASTER_PORT select_data/high_similarity_ddp.py
}


# # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT src/pkt/FSFT.py
# # CUDA_VISIBLE_DEVICES=4,5,6,7 
# torchrun --nproc_per_node=$REQUIRED_GPUS --master_port=$MASTER_PORT src/pkt/FSFT_withAilgn.py
# !/bin/bash

# Constant parameters
BACKEND="vllm"
# MODEL="/data/models/Qwen1.5-72B-Chat"
# Other model options are commented out, uncomment as needed
MODEL="/data/models/Qwen1.5-110B-Chat"
# MODEL="/data/models/DeepSeek-V2-Chat"
# MODEL="/data/models/Yi-1.5-34B-Chat"
N=1
NUM_PROMPTS=10
SEED=1024
MAX_MODEL_LEN=1024
DTYPE="float16"
GPU_MEMORY_UTILIZATION=0.9
DEVICE="cuda"
ENABLE_PREFIX_CACHING="--enable-prefix-caching"

# Create a directory for logs if it doesn't exist
mkdir -p /home/node-user/scripts/logs/Qwen-110B-new

# Loop over tensor-parallel sizes
for TP_SIZE in 2 4 8; do
    for INPUT_LEN in 2048 4096; do
        for OUTPUT_LEN in 128 256; do
            for BATCH_SIZE in 16 32 64; do
                for ENFORCE_EAGER in "--enforce-eager"; do
                    EAGER_STATUS=$(if [ -z "$ENFORCE_EAGER" ]; then echo "no_eager"; else echo "with_eager"; fi)
                    LOG_FILE="/home/node-user/scripts/logs/Qwen-110B-new/benchmark_${DTYPE}_tp${TP_SIZE}_${INPUT_LEN}_${OUTPUT_LEN}_${BATCH_SIZE}_${EAGER_STATUS}.log"
                    
                    python benchmark_throughput.py \
                        --backend $BACKEND \
                        --input-len $INPUT_LEN \
                        --output-len $OUTPUT_LEN \
                        --model $MODEL \
                        --tensor-parallel-size $TP_SIZE \
                        --n $N \
                        --num-prompts $NUM_PROMPTS \
                        --seed $SEED \
                        --max-model-len $MAX_MODEL_LEN \
                        --dtype $DTYPE \
                        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                        $ENFORCE_EAGER \
                        --device $DEVICE \
                        $ENABLE_PREFIX_CACHING \
                        --batch-size $BATCH_SIZE > $LOG_FILE 2>&1
                done
            done
        done
    done
done


# python benchmark_throughput.py \
#   --backend vllm \
#   --input-len 512 \
#   --output-len 128 \
#   --model "/data/models/Qwen1.5-110B-Chat" \
#   --tensor-parallel-size 2 \
#   --n 1 \
#   --num-prompts 10 \
#   --seed 1024 \
#   --max-model-len 1024 \
#   --dtype float16 \
#   --gpu-memory-utilization 0.9 \
#   --device cuda \
#   --enable-prefix-caching \
#   --batch-size 1
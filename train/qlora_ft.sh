# Define environment
export ACCELERATE_USE_FSDP=1
export TOKENIZERS_PARALLELISM=false
export GPU_NUM=4
export MODEL="meta-llama/Llama-3.2-3B-Instruct"
export TASK="pi_emergent" # ["npc", "pi_emergent", "pi_canceled"]
export LORA=true
export QUANTIZATION=true
export DATA_PATH="ccpt_.csv"

# Run training
sr $GPU_NUM 48 torchrun --nproc_per_node $GPU_NUM --nnodes 1 --master_port=$((10000 + RANDOM % 90000)) ./train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --task_type $TASK \
    --output_dir model_params/qlora/$MODEL/$TASK \
    --lora $LORA \
    --quantization $QUANTIZATION \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --fp16 true \
    --gradient_checkpointing true
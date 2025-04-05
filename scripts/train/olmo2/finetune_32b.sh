python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 8 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name olmo2_32b_sft \
    --model_name_or_path allenai/OLMo-2-0325-32B \
    --model_revision main \
    --tokenizer_name allenai/OLMo-2-0325-32B \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --add_bos \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --reduce_loss sum \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 8
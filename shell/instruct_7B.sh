echo $1, $2
seed=$2
output_dir="output13B_sample1000_epoch20"
# base_model="/multimedia-nfs/mahajiag/llama_weights/7B/pytorch"
base_model="/multimedia-nfs/mahajiag/llama_weights/13B/models--openlm-research--open_llama_13b/snapshots/b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8"
train_data="./data/movie/train.json"
val_data="./data/movie/valid.json"
instruction_model="/multimedia-nfs/mahajiag/TALLRec/output13B"
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 1000
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir $output_dir \
                    --batch_size 8 \
                    --micro_batch_size 8 \
                    --num_epochs 20 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2 \
                    --logging_steps 10 \
                    --eval_steps 30
        done
    done
done


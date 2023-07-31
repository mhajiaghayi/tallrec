CUDA_ID="4,5,6,7"
path="output13B_sample1000_epoch20"
base_model="/multimedia-nfs/mahajiag/llama_weights/13B/models--openlm-research--open_llama_13b/snapshots/b6d7fde8392250730d24cc2fcfa3b7e5f9a03ce8"
test_data="./data/movie/test.json"
# for path in $model_path
# do
echo $path
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
    --base_model $base_model \
    --lora_weights $path \
    --test_data_path $test_data \
    --result_json_data $2.json
# done

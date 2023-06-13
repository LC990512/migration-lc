#lc
#--choose_model: bert或roberta\
#--model_name_or_path bert-base-uncased或roberta-base \

# test 
MODELPATH=202303262103_model3/checkpoint-400
TESTPATH=test_case/attention_test_case/attention_test_3.json
CUDA_VISIBLE_DEVICES=0 python src/codeBert/main_cosine.py \
  --choose_model bert \
  --model_name_or_path bert-base-uncased \
  --resume_from_checkpoint ${MODELPATH} \
  --learning_rate 5e-5 \
  --num_train_epochs 15 \
  --output_dir model/inference/model_new_env/craftdroid/\
  --per_device_eval_batch_size=64 \
  --per_device_train_batch_size=64 \
  --overwrite_output \
  --save_strategy steps \
  --save_steps 1000 \
  --do_predict --test_file ${TESTPATH} \
  --mask_index -1 \
  --mask_value -1



# train
# CUDA_VISIBLE_DEVICES=0 python src/codeBert/main_cosine.py \
#   --choose_model roberta \
#   --model_name_or_path roberta-base \
#   --do_train --train_file test_case/roberta_test_case/train_1_data_202303122004_rev.jsonl \
#   --learning_rate 5e-5 \
#   --num_train_epochs 50 \
#   --output_dir model/semifinder/model_new_env/ \
#   --per_device_eval_batch_size=64 \
#   --per_device_train_batch_size=64 \
#   --overwrite_output \
#   --save_strategy steps \
#   --save_steps 400 \
#   --do_eval --validation_file test_case/roberta_test_case/test_1_data_202303220000.jsonl  \
#   --do_predict --test_file test_case/roberta_test_case/test_1_data_202303220000.jsonl \
#   --mask_index -1 \
#   --mask_value -1 \
#   --load_plm_path model/model_roberta \
#   --load_cfg_path model/model_roberta
PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 

F:\anaconda3\envs\stabledif\python.exe E:\yaokui\LLM\chatglm-6b\fine_tune\chatglm_ptuning.py  --do_train --train_file SpamClassify/train.json  --validation_file SpamClassify/dev.json --prompt_column  content --response_column label --overwrite_cache --model_name_or_path ..\\..\\chatglm-6b --output_dir output/spamclassify-chatglm-6b-pt-4-2e-2 --overwrite_output_dir --max_source_length 64 --max_target_length 64 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --predict_with_generate --max_steps 3000 --logging_steps 10 --save_steps 1000 --learning_rate 2e-2 --pre_seq_len 4 --quantization_bit 8

python chatglm_ptuning.py  --do_train --train_file SpamClassify/train.json  --validation_file SpamClassify/dev.json --prompt_column  content --response_column label --overwrite_cache --model_name_or_path ..\\..\\chatglm-6b --output_dir output/spamclassify-chatglm-6b-pt-4-2e-2 --overwrite_output_dir --max_source_length 64 --max_target_length 64 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --predict_with_generate --max_steps 300 --logging_steps 10 --save_steps 100 --learning_rate 2e-2 --pre_seq_len 128

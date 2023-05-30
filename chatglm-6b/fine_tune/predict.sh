PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 python3 

F:\anaconda3\envs\stabledif\python.exe E:\yaokui\LLM\chatglm-6b\fine_tune\chatglm_finetune.py  --do_predict --validation_file AdvertiseGen/dev.json --test_file AdvertiseGen/dev.json --overwrite_cache --prompt_column content --response_column summary --model_name_or_path ..\..\chatglm-6b --ptuning_checkpoint ./output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000 --output_dir ./output/adgen-chatglm-6b-pt-128-2e-2 --overwrite_output_dir --max_source_length 64 --max_target_length 64 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 128 --quantization_bit 8

F:\anaconda3\envs\stabledif\python.exe E:\yaokui\LLM\chatglm-6b\fine_tune\chatglm_ptuning.py  --do_predict --validation_file SpamClassify/dev.json --test_file SpamClassify/dev.json --overwrite_cache --prompt_column content --response_column summary --model_name_or_path ..\..\chatglm-6b --ptuning_checkpoint ./output/spamclassify-chatglm-6b-pt-128-2e-2/checkpoint-3000 --output_dir ./output/spamclassify-chatglm-6b-pt-128-2e-2 --overwrite_output_dir --max_source_length 64 --max_target_length 64 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 4 --quantization_bit 8


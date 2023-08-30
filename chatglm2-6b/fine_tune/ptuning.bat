@echo off

set PRE_SEQ_LEN=128
set LR=2e-2
set NUM_GPUS=1
set MAX_STEP=3000
set save_steps=100

set MODEL_PATH=D:\code\chatglm2-6b-32k

set TRAIN_FILE=role/train.json
set DEV_FILE=role/dev.json
set PROMPT_COLUMN=question
set RESPONSE_COLUMN=answer
set OUT_PATH=output/role-chatglm2-32k-6b-pt-%PRE_SEQ_LEN%-%LR%


REM torchrun --standalone --nnodes=1 --nproc-per-node=%NUM_GPUS%

python main.py ^
    --do_train ^
    --train_file %TRAIN_FILE% ^
    --validation_file %DEV_FILE% ^
    --preprocessing_num_workers 10 ^
    --prompt_column %PROMPT_COLUMN% ^
    --response_column %RESPONSE_COLUMN% ^
    --overwrite_cache ^
    --model_name_or_path %MODEL_PATH% ^
    --output_dir %OUT_PATH% ^
    --overwrite_output_dir ^
    --max_source_length 128 ^
    --max_target_length %PRE_SEQ_LEN% ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps %MAX_STEP% ^
    --logging_steps 10 ^
    --save_steps %save_steps% ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN%
    REM --quantization_bit 4


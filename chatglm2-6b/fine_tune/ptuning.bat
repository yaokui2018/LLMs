@echo off

set PRE_SEQ_LEN=1600
set LR=2e-2
set NUM_GPUS=1
set MAX_STEP=3000

set MODEL_PATH=..\\..\\chatglm2-6b

set TRAIN_FILE=walkthough/train.json
set DEV_FILE=walkthough/dev.json
set PROMPT_COLUMN=title
set RESPONSE_COLUMN=content
set OUT_PATH=output/walkthough-chatglm2-6b-pt-%PRE_SEQ_LEN%-%LR%


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
    --save_steps 1000 ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN% ^
    --quantization_bit 4


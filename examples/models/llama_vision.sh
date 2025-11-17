export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes=4 --main_process_port 12399 -m lmms_eval \
    --model llama_vision \
    --model_args pretrained=unsloth/Llama-3.2-11B-Vision-Instruct \
    --tasks mme \
    --batch_size 1
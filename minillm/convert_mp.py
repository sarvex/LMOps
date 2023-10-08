#coding:utf-8
import torch
import argparse
import os
from transformers import AutoModelForCausalLM
from transformers import (
    decrease_mp_opt, increase_mp_opt,
    decrease_mp_gptj, increase_mp_gptj,
    decrease_mp_llama, increase_mp_llama,
    decrease_mp_gpt2, increase_mp_gpt2,
)

func_map = {
    "opt": (decrease_mp_opt, increase_mp_opt),
    "gptj": (decrease_mp_gptj, increase_mp_gptj),
    "llama": (decrease_mp_llama, increase_mp_llama),
    "gpt2": (decrease_mp_gpt2, increase_mp_gpt2),
}


def main():
    parser = argparse.ArgumentParser("Transform huggingface checkpoints to megatron+deepspeed checkpoints")

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--model_type", type=str, default="opt")
    parser.add_argument("--source_mp_size", type=int, default=1)
    parser.add_argument("--target_mp_size", type=int, default=2)
    # parser.add_argument("--save_path", type=str)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--exist_ok", action="store_true")

    args = parser.parse_args()

    decrease_mp, increase_mp = func_map[args.model_type]

    if args.source_mp_size == 1:
        assert args.target_mp_size > args.source_mp_size
        args.save_path = os.path.join(args.input_path, f"mp{args.target_mp_size}")
        assert args.exist_ok or not any(
            os.path.exists(
                os.path.join(args.save_path, f"pytorch_model_{i}.bin")
            )
            for i in range(args.target_mp_size)
        )
        os.makedirs(args.save_path, exist_ok=True)
        model_hf = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16).state_dict()
        d_list = increase_mp(model_hf, args.target_mp_size, half=args.half)
        for i, d in enumerate(d_list):
            torch.save(d, os.path.join(args.save_path, f"pytorch_model_{i}.bin"))
    elif args.target_mp_size == 1:
        assert args.source_mp_size > args.target_mp_size
        args.save_path = args.input_path
        assert args.exist_ok or not os.path.exists(os.path.join(args.save_path, "pytorch_model.bin"))
        ckpt_path = os.path.join(args.input_path, f"mp{args.source_mp_size}")
        d_list = [torch.load(os.path.join(ckpt_path, f"pytorch_model_{i}.bin"), map_location="cpu") for i in range(args.source_mp_size)]
        d = decrease_mp(d_list, half=args.half)
        torch.save(d, os.path.join(args.save_path, "pytorch_model.bin"))
    else:
        args.save_path = os.path.join(args.input_path, f"mp{args.target_mp_size}")
        assert args.exist_ok or not any(
            os.path.exists(
                os.path.join(args.save_path, f"pytorch_model_{i}.bin")
            )
            for i in range(args.target_mp_size)
        )

        ckpt_path = os.path.join(args.input_path, f"mp{args.source_mp_size}")
        d_list = [torch.load(os.path.join(ckpt_path, f"pytorch_model_{i}.bin"), map_location="cpu") for i in range(args.source_mp_size)]
        d = decrease_mp(d_list, half=args.half)

        torch.save(d, os.path.join(args.input_path, "pytorch_model.bin"))

        os.makedirs(args.save_path, exist_ok=True)
        model_hf = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16).state_dict()
        d_list = increase_mp(model_hf, args.target_mp_size, half=args.half)
        for i, d in enumerate(d_list):
            torch.save(d, os.path.join(args.save_path, f"pytorch_model_{i}.bin"))
        
        
        
    
if __name__ == '__main__':
    main()

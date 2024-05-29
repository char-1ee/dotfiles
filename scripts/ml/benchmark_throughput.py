"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase, Qwen2Config)

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

GIGABYTE = 1024**3
MEGABYTE = 1024 * 1024
N_WARMUPS = 2

def run_vllm(
    # requests: List[Tuple[str, int, int]],
    prompt_token_ids: List[List[int]],
    input_len: int,
    output_len: int,
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    # max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=input_len + output_len,
        # max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    
    num_requests = len(prompt_token_ids)
    
    # Warm up
    for _ in range(N_WARMUPS):
        for i in range(num_requests):
            sampling_params = SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            )
            llm._add_request(
                prompt=None,
                prompt_token_ids=prompt_token_ids[i],
                sampling_params=sampling_params,
            )
        llm._run_engine(use_tqdm=True)
        
    # Benchmark
    for i in range(num_requests):
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        llm._add_request(
            prompt=None,
            prompt_token_ids=prompt_token_ids[i],
            sampling_params=sampling_params,
        )

    torch.cuda.synchronize()
    start = time.perf_counter() # timing without add requests procedure
    # FIXME(woosuk): Do not use internal method.
    output = llm._run_engine(use_tqdm=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start, output


def get_current_device():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Using device: {device}")
    return device


def prepare_data(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_current_device())
    return input_ids


def print_details_info(model_config, args, whole_end2end, total_token_num):
    msg: str = ""

    if torch.distributed.get_rank() == 0:
        msg += "-------Perf Summary-------\n"
        whole_avg_latency = whole_end2end / (total_token_num)
        num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
        num_parameters = (
            num_layers * model_config.hidden_size * model_config.hidden_size * 12
        )
        if args.dtype in ["float16", "bf16"]:
            num_bytes = 2
        else:
            num_bytes = 4

        msg += f"Whole batch end2end time: {whole_end2end * 1000:.2f} ms\n"
        msg += f"Whole batch per token latency: {whole_avg_latency * 1000:.2f} ms\n"
        msg += f"Total token number: {total_token_num}\n"
        msg += f"Throughput: {total_token_num / whole_end2end:.2f} tokens/s\n"
        msg += f"Flops: {num_parameters * num_bytes / whole_avg_latency / 1e12:.2f} TFLOPS\n"

    msg += f"-------Memory Summary Device:{get_current_device()}-------\n"
    msg += f"Max memory allocated: {torch.cuda.max_memory_allocated(device=get_current_device()) / GIGABYTE:.2f} GB\n"
    msg += f"Max memory reserved: {torch.cuda.memory_reserved(device=get_current_device()) / GIGABYTE:.2f} GB\n"

    print(msg)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    # Synthesize a prompt with the given input length.
    # prompt = "hi" * (args.input_len - 1)
    # requests = [(prompt, args.input_len, args.output_len)
    #             for _ in range(args.num_prompts)]
    
    input_ids = prepare_data(args.batch_size, args.input_len).tolist()

    if args.backend == "vllm":
        elapsed_time, output = run_vllm(
            # requests, 
            input_ids,
            args.input_len,
            args.output_len,
            args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, 
            # args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.gpu_memory_utilization,
            args.download_dir)
    else:
        raise ValueError(f"Unknown backend: {args.backeznd}")
    
    # Profiling results
    # total_num_tokens = sum(prompt_len + output_len
    #                        for _, prompt_len, output_len in len(requests))
    print("================================================")
    total_num_tokens = sum([len(out.outputs[0].token_ids) for out in output])
    print(f"Throughput: {len(input_ids) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s"
          f" ({total_num_tokens} tokens in {elapsed_time:.2f} seconds)"
        )
    print("================================================")
    
    # TODO: change config 
    # config = Qwen2Config()
    config = DeepSeekV2Config()
    print_details_info(config, args, elapsed_time, total_num_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")

    main(args)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2" # TODO
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def test_model_inference(batch_sizes, seq_lengths):
    results = []
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            # Generate random inputs
            inputs = tokenizer(["X" * seq_length] * batch_size, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = inputs["input_ids"]

            # Warm up
            with torch.no_grad():
                _ = model(input_ids)

            # Timing inference
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            elapsed_time = time.time() - start_time

            # Calculate latency and throughput
            latency = elapsed_time / batch_size
            throughput = batch_size / elapsed_time

            results.append((batch_size, seq_length, latency, throughput))
            print(f"Batch size: {batch_size}, Seq Length: {seq_length}, Latency: {latency:.6f} sec, Throughput: {throughput:.2f} inferences/sec")

    return results

batch_sizes = [1, 2, 4]  # Add more batch sizes as needed
seq_lengths = [4096, 8192, 16384, 32768]  # Sequence lengths as required
test_model_inference(batch_sizes, seq_lengths)

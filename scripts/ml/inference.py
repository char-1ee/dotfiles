import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name_or_path = "/home/node-user/models/CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval().to(device)

        
def prepare_data(batch_size: int = 4, seq_len: int = 1024):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    data = dict(input_ids=input_ids, attention_mask=attention_mask)
    return data


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
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time

            # Calculate latency and throughput
            latency = elapsed_time / batch_size
            throughput = batch_size / elapsed_time

            results.append((batch_size, seq_length, latency, throughput))
            print(f"Batch size: {batch_size}, Seq Length: {seq_length}, Latency: {latency:.6f} sec, Throughput: {throughput:.2f} inferences/sec")

    return results

if __name__ == "__main__":
    batch_sizes = [1]
    seq_lengths = [4096]
    # seq_lengths = [4096, 8192, 16384, 32768]
    test_model_inference(batch_sizes, seq_lengths)
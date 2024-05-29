from huggingface_hub import snapshot_download

model_ids = [
    "01-ai/Yi-1.5-34B-Chat",
    "CohereForAI/c4ai-command-r-v01",
    "CohereForAI/c4ai-command-r-plus-4bit",
    "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-110B-Chat",
    "deepseek-ai/DeepSeek-V2-Chat"
]

parent_dir = "/home/node-user/models"

max_workers = 8

for model_id in model_ids:
    local_dir = f"{parent_dir}/{model_id.split('/')[-1]}"
    print(f"Starting download for: {model_id}")
    snapshot_download(repo_id=model_id, local_dir=local_dir, max_workers=max_workers)
    print(f"Model downloaded to: {local_dir}")

print("All models have been downloaded.")

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download gpt2 --local-dir gpt2
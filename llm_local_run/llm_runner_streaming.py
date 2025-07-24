# Streaming Version - Fastest Perceived Response Time
# !pip install llama-cpp-python

from llama_cpp import Llama
import time
import os
import multiprocessing

# Auto-detect CPU cores for optimal threading
cpu_cores = multiprocessing.cpu_count()
optimal_threads = min(cpu_cores, 16)

# Optimize CPU usage
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)

print(f"🚀 Streaming LLM Runner - Detected {cpu_cores} CPU cores, using {optimal_threads} threads")

# Load model with maximum speed settings
start_time = time.time()
llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-0.6B-GGUF",
	filename="Qwen3-0.6B-UD-Q8_K_XL.gguf",
	n_ctx=1024,  # Smaller context for maximum speed
	n_batch=2048,  # Large batch for throughput
	n_threads=optimal_threads,
	#n_gpu_layers=-1,  # Uncomment if you have GPU
	f16_kv=True,
	use_mmap=True,
	use_mlock=True,
	verbose=False
)
load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f} seconds")

# Streaming inference for fastest perceived response
print("\n🎯 Starting streaming inference...")
start_time = time.time()

# Create streaming response
stream = llm.create_chat_completion(
	messages=[
		{
			"role": "system",
			"content": "You are an HR assistant. Break down complex tasks into smaller actionable steps."
		},
		{
			"role": "user",
			"content": "Create and manage fillable forms for onboarding and compliance."
		}
	],
	max_tokens=300,  # Reasonable length for speed
	temperature=0.3,  # Lower temperature for faster, more focused responses
	top_p=0.8,
	stream=True  # Enable streaming for immediate response
)

# Print response as it streams
full_response = ""
token_count = 0
first_token_time = None

print("📝 Response (streaming):")
print("-" * 50)

for chunk in stream:
	if 'choices' in chunk and len(chunk['choices']) > 0:
		delta = chunk['choices'][0].get('delta', {})
		if 'content' in delta:
			content = delta['content']
			print(content, end='', flush=True)
			full_response += content
			token_count += 1
			
			# Record time to first token
			if first_token_time is None:
				first_token_time = time.time() - start_time

end_time = time.time()
total_time = end_time - start_time

print("\n" + "-" * 50)
print(f"⚡ Performance Metrics:")
print(f"   First token: {first_token_time:.2f}s")
print(f"   Total time: {total_time:.2f}s")
print(f"   Tokens generated: {token_count}")
print(f"   Speed: {token_count/total_time:.1f} tokens/second")

# Save response
with open("streaming_response.txt", "w") as f:
	f.write(full_response)

print(f"💾 Response saved to streaming_response.txt") 
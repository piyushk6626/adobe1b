# !pip install llama-cpp-python

from llama_cpp import Llama
import time
import os
import multiprocessing

# Auto-detect CPU cores for optimal threading
cpu_cores = multiprocessing.cpu_count()
optimal_threads = min(cpu_cores, 16)  # Cap at 16 for diminishing returns

# Optimize CPU usage
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)

print(f"Detected {cpu_cores} CPU cores, using {optimal_threads} threads")

start_time = time.time()
llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-0.6B-GGUF",
	filename="Qwen3-0.6B-UD-Q8_K_XL.gguf",
	n_ctx=2048,  # Reduced context window for faster processing
	n_batch=1024,  # Larger batch size for better GPU/CPU utilization
	n_threads=optimal_threads,  # Auto-optimized thread count
	#n_gpu_layers=-1,  # Use GPU acceleration (uncomment if you have GPU)
	f16_kv=True,  # Use half precision for KV cache (saves memory)
	use_mmap=True,  # Memory mapping for faster model loading
	use_mlock=True,  # Lock model in memory to prevent swapping
	verbose=False
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
start_time = time.time()
response = llm.create_chat_completion(
	messages = [
		{
			"role": "system",
			"content": """
            you are A HR and your 
            #### Primary Objective:\nRetrieve strategic information related to talent management, organizational effectiveness, employee relations, 
            # and HR metrics to align human resources strategy with business unit objectives.\n\n
            # #### Core Information to Retrieve:\n-   
            # **Strategic Documents:** Phrases like \"workforce planning,\" \"succession planning,\" \"talent review,\" \"organizational design,\"
            #  \"headcount planning,\" \"skills gap analysis,\" \"business unit goals.\"\n-   **Employee Performance Data:** Performance improvement plans (PIPs),
            #  performance review summaries, employee goal sheets, 360-degree feedback results, lists of high-potential employees.\n-   
            # **Employee Relations & Engagement:** \"Employee engagement survey\" results, exit interview analysis, grievance summaries, disciplinary action 
            # reports, \"culture assessment\" documents.\n-   **Compensation & Talent Retention:** \"Salary bands,\" \"compensation philosophy,\" 
            # \"market rate analysis,\" \"bonus structure,\" \"retention risk,\" \"turnover rate,\" \"promotion data.\"\n-   
            # **Change Management:** Documents related to \"restructuring,\" \"merger and acquisition\" (M&A) diligence, \"change management plan,\" 
            # \"reduction in force\" (RIF) planning.\n-   **HR Policies & Compliance:** \"EEO\" reports, \"harassment policy,\" \"FMLA trends,\"
            #  \"ADA accommodations,\" \"code of conduct,\" investigation protocols.\n-   **Leadership & Development:** \"Leadership development program,\"
            #  \"competency models,\" \"training needs analysis.\"\n-   
            # **Metrics & Reports:** HR dashboards, attrition reports, diversity and inclusion metrics, recruiting funnel statistics, 
            # cost-per-hire data.\n\n#### Information to Ignore (Noise):\n-   Individual employee payroll stubs or routine benefits enrollment forms.\n-  
            #  Daily employee timesheets or attendance logs.\n-   Individual expense reports not related to a policy investigation.\n-   IT support tickets 
            # and system maintenance notifications.\n-   Marketing materials, sales reports, or product development roadmaps.\n-   Routine facilities management 
            # requests or building maintenance schedules.\n-   Company social event planning details (e.g., catering menus, venue selection).\n-  
            #  Highly technical engineering documents or software code.
            # tour task is to brake this task into smaller one"""
		},
        {
			"role": "user",
			"content": "Create and manage fillable forms for onboarding and compliance."
		}
	],
	max_tokens=2048,  # Limit response length for faster generation
	temperature=0.7,  # Slightly reduce for faster sampling
	top_p=0.9,  # Nucleus sampling for efficiency
	repeat_penalty=1.1,  # Prevent repetition
	top_k=40  # Limit vocabulary consideration for speed
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
with open("response.txt", "w") as f:
    f.write(response["choices"][0]["message"]["content"])

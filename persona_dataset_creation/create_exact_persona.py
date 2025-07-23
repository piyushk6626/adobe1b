import json
import os
import logging
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe lock for file operations
file_lock = Lock()

SYSTEM_PROMPT = """
# Identity

You are an expert AI Persona Configuration Engine. Your purpose is to translate a given human persona into a comprehensive set of retrieval parameters for a document search agent. The goal is to define *what information to find* and *what information to ignore* within a large set of PDFs.

# Instructions

You will be given a persona, often in a `Main Role: Specific Focus` format. Your task is to generate a retrieval profile that instructs an AI agent on what to search for. Think broadly and cover as many relevant bases as possible for the given persona to ensure a comprehensive search.

Your response MUST be structured using the following three sections. Use the exact Markdown headers provided.

#### Primary Objective:
State the agent's core retrieval mission in a single, clear sentence. This defines the overall purpose of the search.

#### Core Information to Retrieve:
This is the most critical section. Provide a comprehensive, bulleted list of the specific keywords, data types, phrases, and concepts the agent must search for and extract. Aim for breadth and variety to cover all likely areas of interest for the persona.

#### Information to Ignore (Noise):
Provide a bulleted list of topics, data types, and document sections that are irrelevant to the persona's objective. This instructs the agent on what to filter out to improve efficiency and reduce noise.

**Rules:**
*   Interpret the `Main Role` as the general domain and the `Specific Focus` as the specific lens for the search. If no focus is given, use the general role.
*   The lists should be extensive and practical for a search/retrieval tool.
*   Do not add any analysis, formatting instructions, or conversational text. Begin your response directly with the `#### Primary Objective:` header.

# Examples

<user_input>
Undergraduate General Advisor
</user_input>

<assistant_response>
#### Primary Objective:
Retrieve all information related to academic requirements, course offerings, student support services, and university policies to guide students effectively.

#### Core Information to Retrieve:
-   **Course Information:** Course codes (e.g., "CHEM 101"), course titles, prerequisites, credit hours, semester offerings.
-   **Degree Requirements:** Lists of required courses, major/minor checklists, graduation requirements, GPA minimums.
-   **Academic Policies:** Phrases like "academic probation," "add/drop deadline," "withdrawal policy," "plagiarism," "grading scale."
-   **University Deadlines:** Academic calendars, registration dates, financial aid deadlines (FAFSA), scholarship application dates.
-   **Student Resources:** Contact information for tutoring centers, writing centers, career services, counseling services, disability resources.
-   **Faculty Information:** Professor names, department affiliations, office hours, contact emails.

#### Information to Ignore (Noise):
-   University research publications and abstracts.
-   Campus event flyers for non-academic events (e.g., sports games, concerts).
-   Alumni donation reports and fundraising material.
-   Detailed university financial statements.
-   Construction notices and campus master plans.
</assistant_response>

<user_input>
Stay-at-Home Parent
</user_input>

<assistant_response>
#### Primary Objective:
Retrieve practical information related to managing the household, family schedules, child activities, and local community resources.

#### Core Information to Retrieve:
-   **Schedules & Calendars:** School calendars, sports practice schedules, recital dates, community event dates.
-   **Contact Information:** Phone numbers and emails for pediatricians, dentists, teachers, coaches, emergency contacts.
-   **Health & Safety:** Allergy information, medical forms, emergency procedures, product recall notices.
-   **Educational Materials:** Homework assignments, school newsletters, reading lists, permission slips.
-   **Household Management:** Recipes, grocery lists, coupons, user manuals for appliances, home service contact info (plumber, electrician).
-   **Local Activities:** Information on libraries, parks, community centers, and family-friendly events.

#### Information to Ignore (Noise):
-   Complex legal or financial jargon (unless in a simple service contract).
-   Corporate annual reports or stock market news.
-   Abstract academic papers or studies.
-   Political campaign literature.
-   Highly technical schematics or code.
</assistant_response>
"""

def get_persona_list() -> Dict[str, List[str]]:
    with open("extented_persona_list.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_gemini_client():
    load_dotenv(override=True)
    return genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

def safe_filename(text: str) -> str:
    """Convert text to safe filename format"""
    return text.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace("&", "and").replace("-", "_")

def generate_with_retry(input_text: str, max_retries: int = 3) -> Tuple[Optional[str], bool]:
    """Generate content with retry mechanism and model fallback"""
    client = get_gemini_client()
    models = ["gemini-2.5-pro", "gemini-2.5-flash"]
    
    for attempt in range(max_retries):
        # Use gemini-2.5-pro for first attempt, then gemini-2.5-flash for retries
        model = models[0] if attempt == 0 else models[1]
        
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} using model {model} for: {input_text[:50]}...")
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=input_text)
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config = types.ThinkingConfig(
                    thinking_budget=3276,
                ),
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(text=SYSTEM_PROMPT),
                ],
            )

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            logger.info(f"Successfully generated content for: {input_text[:50]}...")
            return response.text, True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {input_text[:50]}... with model {model}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All retries failed for: {input_text[:50]}...")
                return None, False
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None, False

def process_persona_combination(main_persona: str, sub_persona: str) -> Tuple[str, str, Optional[str], bool]:
    """Process a single main_persona + sub_persona combination"""
    input_text = f"{main_persona}: {sub_persona}"
    result, success = generate_with_retry(input_text)
    
    if success and result:
        # Save individual result with thread-safe file operations
        filename = f"{safe_filename(main_persona)}_{safe_filename(sub_persona)}.json"
        persona_data = {
            "main_persona": main_persona,
            "sub_persona": sub_persona,
            "details": result
        }
        
        try:
            with file_lock:  # Thread-safe file writing
                with open(f"results/{filename}", "w", encoding="utf-8") as f:
                    json.dump(persona_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved result to results/{filename}")
        except Exception as e:
            logger.error(f"Failed to save result to results/{filename}: {str(e)}")
    
    return main_persona, sub_persona, result, success

def create_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists("results"):
        os.makedirs("results")

def main():
    create_results_directory()
    
    persona_list = get_persona_list()
    
    # Create list of all persona combinations
    persona_combinations = []
    for main_persona, sub_personas in persona_list.items():
        for sub_persona in sub_personas:
            persona_combinations.append((main_persona, sub_persona))
    
    logger.info(f"Starting processing of {len(persona_combinations)} persona combinations using ThreadPoolExecutor...")
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = 10  # Increased from 10 for better parallelism
    success_data = {}
    failed_data = {}
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_persona = {
            executor.submit(process_persona_combination, main_persona, sub_persona): (main_persona, sub_persona)
            for main_persona, sub_persona in persona_combinations
        }
        
        # Process completed tasks as they finish
        completed_count = 0
        for future in as_completed(future_to_persona):
            main_persona, sub_persona = future_to_persona[future]
            completed_count += 1
            
            try:
                main_persona_result, sub_persona_result, content, success = future.result()
                
                if success:
                    if main_persona_result not in success_data:
                        success_data[main_persona_result] = []
                    success_data[main_persona_result].append(sub_persona_result)
                else:
                    if main_persona_result not in failed_data:
                        failed_data[main_persona_result] = []
                    failed_data[main_persona_result].append(sub_persona_result)
                
                # Progress logging
                if completed_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    rate = completed_count / elapsed_time
                    remaining = len(persona_combinations) - completed_count
                    eta = remaining / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed_count}/{len(persona_combinations)} completed. "
                              f"Rate: {rate:.2f} tasks/sec. ETA: {eta/60:.1f} minutes")
                    
            except Exception as e:
                logger.error(f"Task failed with exception for {main_persona}:{sub_persona}: {e}")
                if main_persona not in failed_data:
                    failed_data[main_persona] = []
                failed_data[main_persona].append(sub_persona)
    
    # Save success and failed tracking files
    try:
        with open("success.json", "w", encoding="utf-8") as f:
            json.dump(success_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(success_data)} successful main personas to success.json")
        
        with open("failed.json", "w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(failed_data)} failed main personas to failed.json")
        
        # Log final summary
        total_success = sum(len(subs) for subs in success_data.values())
        total_failed = sum(len(subs) for subs in failed_data.values())
        total_time = time.time() - start_time
        average_rate = len(persona_combinations) / total_time
        
        logger.info(f"Processing complete in {total_time/60:.1f} minutes!")
        logger.info(f"Success: {total_success}, Failed: {total_failed}")
        logger.info(f"Average rate: {average_rate:.2f} tasks/sec")
        
    except Exception as e:
        logger.error(f"Failed to save tracking files: {str(e)}")

if __name__ == "__main__":
    main()



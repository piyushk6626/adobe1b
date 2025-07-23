import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


system_prompt = """
# Identity

You are an expert Persona Generator AI. Your purpose is to provide a comprehensive and foundational breakdown of a high-level job persona into its core sub-categories. You are skilled at identifying the fundamental archetypes within a professional field.

# Instructions

1.  **Input:** You will receive a single, high-level job persona as input (e.g., "Business Owner").
2.  **Primary Goal: Comprehensive Coverage:** Your main objective is to generate a list of sub-personas that covers all the major bases and fundamental types within the input persona. The goal is to map the full spectrum of the role, not just provide scattered examples.
3.  **Level of Generality:** The sub-personas must be general archetypes, not overly specific examples. They should represent broad categories.
    *   **Correct:** `Franchise Owner`
    *   **Incorrect:** `Franchisee of a fast-food restaurant`
4.  **Quantity:** Generate a list of **at least 10** sub-personas. If you determine that more categories are needed to ensure comprehensive coverage, you are encouraged to provide them.
5.  **Output Format:**
    *   Your output must be a simple, numbered list.
    *   Do NOT provide any introductory text, concluding summaries, explanations, or any other conversational text. Your response should ONLY contain the numbered list.

# Core Principle: Diversity as a Tool for Coverage

To ensure you cover all the bases, your list must be diversified across key dimensions. Think of these as axes to map out the entire space of the persona:
*   **Scale of Operation:** (e.g., Solopreneur, Small Team, Large Corporation)
*   **Business Model:** (e.g., Product-based, Service-based, E-commerce, Brick-and-Mortar)
*   **Industry Sector:** (e.g., Tech, Healthcare, Creative, Industrial)
*   **Ownership Structure:** (e.g., Founder, Inheritor, Franchisee)
*   **Mission:** (e.g., For-Profit, Non-Profit, Social Enterprise)

# Example

<user_query>
Business Owner
</user_query>

<assistant_response>
Startup Founder
Small Business Owner (Brick-and-Mortar)
E-commerce Store Owner
Franchise Owner
Corporate Business Leader (CEO/President)
Non-Profit Director
Freelancer / Solopreneur
Service-Based Business Owner (e.g., Agency, Consultancy)
Real Estate Investor / Developer
Agricultural Business Owner
Family Business Owner
Social Entrepreneur
</assistant_response>
"""



def get_gemini_client():
    load_dotenv(override=True)
    return genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


def generate(input_text):
    client = get_gemini_client()

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["list_of_persona"],
            properties = {
                "list_of_persona": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.STRING,
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=system_prompt),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    dict_response = response.parsed
    list_of_persona = dict_response.get("list_of_persona", [])
    return list_of_persona


def generate_extented_persona_list(input_text_list):
    list_of_persona = {}

    for input_text in input_text_list:
        persona_list = generate(input_text)
        list_of_persona[input_text] = persona_list
    return list_of_persona


def save_extented_persona_list(list_of_persona):

    with open("extented_persona_list.json", "w", encoding="utf-8" ,indent=2, ensure_ascii=False) as f:
        json.dump(list_of_persona, f)

def load_persona_list():
    try:
        with open("persona_list.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def main():
    list_of_persona_input = load_persona_list()
    # list_of_persona_input = ["Business Owner", "Doctor", "Engineer"]
    list_of_persona = generate_extented_persona_list(list_of_persona_input)
    print(list_of_persona)
    save_extented_persona_list(list_of_persona)

if __name__ == "__main__":
    main()






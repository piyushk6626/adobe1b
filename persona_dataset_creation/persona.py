import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


input_text_1 = """### 🧾 **Business & Finance**

1. **Invoice** – Billing documents showing itemized charges.
2. **Quotation/Estimate** – Price quotes for services or products.
3. **Purchase Order (PO)** – Orders placed by customers to suppliers.
4. **Receipt** – Proof of payment received.
5. **Balance Sheet / Financial Statement** – Summaries of financial data.
6. **Tax Forms** – Like Form 16, W-9, 1099, GST returns.
7. **Bank Statements** – Monthly records of account activity.

---"""

input_text_2 = """


### 📚 **Educational & Academic**

1. **Textbooks** – Complete academic books.
2. **Research Papers / Journals** – Academic articles, IEEE, arXiv.
3. **Lecture Notes / Handouts**
4. **Thesis / Dissertation**
5. **Student Report Cards / Mark Sheets**
6. **Syllabus / Curriculum PDFs**

---"""

input_text_3 = """
### 📄 **Legal & Compliance**

1. **Contracts / Agreements** – Service, lease, NDA, employment.
2. **Terms & Conditions**
3. **Privacy Policy**
4. **Court Documents** – Judgments, case files.
5. **Government Forms** – PAN application, passport forms, voter ID.

---"""

input_text_4 = """
### 📢 **Marketing & Design**

1. **Brochure** – Multi-page promotional document.
2. **Flyer / Pamphlet** – One-pagers for events or offers.
3. **Catalog** – Product showcase.
4. **Portfolio** – For designers, artists, photographers.
5. **Press Release**

---"""

input_text_5 = """
### 🧑‍💼 **Corporate & HR**

1. **Resume / CV**
2. **Offer Letter / Appointment Letter**
3. **Employee Handbook / Policies**
4. **Onboarding Docs / Forms**
5. **Appraisal Reports**

---"""

input_text_6 = """
### 📘 **Product & Tech**

1. **User Manual / Guide**
2. **Product Datasheet**
3. **API Documentation (PDF version)**
4. **Software License Agreement**

---"""

input_text_7 = """
### ✈️ **Travel & Lifestyle**

1. **Itinerary / Travel Plan**
2. **Boarding Pass / E-Ticket**
3. **Hotel Booking Confirmation**
4. **Travel Insurance PDF**
5. **Tourist Brochure / Guidebook**

---"""

input_text_8 = """
### 🏠 **Real Estate & Utility**

1. **Rent Agreement**
2. **Electricity / Water / Gas Bill**
3. **Property Brochure**
4. **Site Plans / Floor Plans**

---"""

input_text_9 = """
### 🏥 **Medical & Health**

1. **Prescription**
2. **Medical Reports / Lab Tests**
3. **Health Insurance Policy**
4. **Vaccination Certificate**

---"""

input_text_10 = """
### 📦 **Logistics & Shipping**

1. **Packing List**
2. **Bill of Lading**
3. **Shipping Label**
4. **Delivery Note / Proof of Delivery**

---"""

base_prompt = """
# Identity

You are an expert Persona Generation Assistant. Your purpose is to analyze lists of document types and identify 10 distinct, high-level user personas who would interact with them. You should think in broad categories like 'Manager', 'Student', or 'Engineer'.

# Instructions

You will be given a category and a list of document types within that category. Your task is to generate a list of exactly 10 distinct, high-level, and general user personas based on that input.

**Rules for Generation:**
*   **Output Format:** Your entire response must be a numbered list from 1 to 10.
*   **Content Simplicity:** For each number, provide **only** a general role or title. The title should be a single phrase or job category (e.g., "Business Owner", "Accountant", "Student").
*   **Strict Negative Constraint:** Do **not** provide any descriptions, explanations, or sentences. Your output for each number must be the role/title ONLY.
*   **Diversity:** Ensure the roles are diverse and represent different general functions (e.g., finance, operations, sales, individual, executive).
*   **Final Output Constraint:** Do not write any text before "1." or after the 10th item.

# Examples

This section provides a clear example of the expected input and the corresponding high-quality output, which is just a list of high-level roles.

<user_input>
### 🧾 **Business & Finance**

1.  **Invoice** – Billing documents showing itemized charges.
2.  **Quotation/Estimate** – Price quotes for services or products.
3.  **Purchase Order (PO)** – Orders placed by customers to suppliers.
4.  **Receipt** – Proof of payment received.
5.  **Balance Sheet / Financial Statement** – Summaries of financial data.
6.  **Tax Forms** – Like Form 16, W-9, 1099, GST returns.
7.  **Bank Statements** – Monthly records of account activity.
</user_input>

<assistant_response>
1. Business Owner
2. Accountant
3. Freelancer
4. Salesperson
5. Manager
6. Financial Analyst
7. Executive
8. Administrator
9. Consultant
10. Customer
</assistant_response>
"""

def get_gemini_client():
    load_dotenv(override=True)
    return genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


def generate(input_text):
    client = get_gemini_client()

    model = "gemini-2.5-pro"
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
            types.Part.from_text(text=base_prompt),
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


def generate_persona_list(input_text_list):
    list_of_persona = []
    for input_text in input_text_list:
        list_of_persona.extend(generate(input_text))
    return list_of_persona

def save_persona_list(list_of_persona):
    with open("persona_list.json", "w") as f:
        json.dump(list_of_persona, f)


if __name__ == "__main__":
    list_of_persona = generate_persona_list([input_text_1, input_text_2, input_text_3, input_text_4, input_text_5, input_text_6, input_text_7, input_text_8, input_text_9, input_text_10])
    print(list_of_persona)
    save_persona_list(list_of_persona)
import os
import json
from groq import Groq # Make sure you have run: pip install groq
from dotenv import load_dotenv
load_dotenv()
# --- 1. CONFIGURATION ---
# IMPORTANT: Replace with your actual Groq API Key

# Alternatively, set it as an environment variable named GROQ_API_KEY
# if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
#     GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
#     if not GROQ_API_KEY:
#         print("ERROR: GROQ_API_KEY not set. Please set it in the script or as an environment variable.")
#         exit()

# --- 2. LOAD AUGMENTED SCHEMA ---
try:
    with open("augmented_schema.json", "r") as f:
        augmented_schema_data = json.load(f)
    # Convert the loaded JSON data back into a pretty-printed string for the prompt
    augmented_schema_string = json.dumps(augmented_schema_data, indent=2)
except FileNotFoundError:
    print("ERROR: augmented_schema.json not found in the current directory.")
    print("Please ensure the file exists and the script is run from the same directory.")
    exit()
except json.JSONDecodeError:
    print("ERROR: augmented_schema.json contains invalid JSON. Please check its format.")
    exit()

# --- 3. CHOOSE LLM & DEFINE USER QUESTION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Recommended models on Groq for this task:
# model_to_use = "llama3-70b-8192" # Most capable, best for complex SQL
model_to_use = "deepseek-r1-distill-llama-70b" # Good balance of capability and speed
# model_to_use = "llama3-8b-8192"   # Faster, good for simpler queries

# ---- VVVV ---- EDIT YOUR QUESTION HERE ---- VVVV ----
user_question = "What is the total quantity of 'Laptop Pro X' sold, excluding returns?"
# Other example questions you can try:
# user_question = "How many unique active customers made a purchase in January 2023?"
# user_question = "List the top 3 product categories by total sales revenue (price_total after cleaning)."
# user_question = "Show customer names and their join dates for customers who joined after 2022 and live in 'New York, NY'."
# user_question = "What was the total discount amount given for orders placed via 'ORD1001'?"
# ---- ^^^^ ---- EDIT YOUR QUESTION HERE ---- ^^^^ ----


# --- 4. CONSTRUCT THE PROMPT ---
# This prompt is crucial for getting good SQL.
prompt = f"""You are an expert PostgreSQL data analyst. Your task is to convert user questions into a single, valid PostgreSQL query based on the provided database schema description.
The schema contains tables and columns with non-obvious 'actual_name's; you MUST use these 'actual_name's in the generated SQL.
Rely on the 'semantic_name' (for your understanding) and 'description' fields to map user questions to the correct database entities.
Pay close attention to data types, relationships, and the 'general_hints' for handling dirty data, data transformations (like date parsing or string cleaning for numbers), and specific business logic (like negative quantity for returns).
Only output the SQL query. Do not include any explanations or markdown formatting before or after the SQL query itself.

Database Schema Description:
{augmented_schema_string}

User Question: {user_question}

PostgreSQL Query:
"""

# --- 5. CALL THE LLM API ---
print(f"--- Asking '{model_to_use}' on Groq to generate SQL for: ---")
print(f"'{user_question}'")
print("--- Waiting for LLM response... ---")

try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user", # Some models respond better if the full context is in one user message
                "content": prompt,
            }
        ],
        model=model_to_use,
        temperature=0.0, # Set to 0.0 for most deterministic SQL output
        # max_tokens=1024, # Adjust if queries get truncated, though the prompt asks for only SQL.
    )

    generated_sql = chat_completion.choices[0].message.content.strip()

    print("\n--- Generated SQL Query ---")
    print(generated_sql)

    print("\n--- Instructions ---")
    print("1. Carefully review the generated SQL query above.")
    print("2. Copy the SQL query.")
    print("3. Go to your Supabase project's SQL Editor.")
    print("4. Paste the SQL query and click 'RUN'.")
    print("5. Check if the results are correct and what you expected.")
    print("   - If it's not working or incorrect, try:")
    print("     a) Modifying the 'user_question' in this script.")
    print("     b) Adjusting the details in 'augmented_schema.json' (especially descriptions and hints).")
    print("     c) Tweaking the main system prompt within this script.")
    print("     d) Trying a different 'model_to_use'.")

except Exception as e:
    print(f"\nAn error occurred while communicating with the Groq API: {e}")
    if "401" in str(e) or "authentication" in str(e).lower():
        print("This might be an API key issue. Please double-check your GROQ_API_KEY.")

print(f"\nScript finished. Using model: {model_to_use}")
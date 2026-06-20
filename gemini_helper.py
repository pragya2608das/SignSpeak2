import os
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------------------
# Load Environment Variables
# -----------------------------------------

load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )

# -----------------------------------------
# Configure Gemini
# -----------------------------------------

genai.configure(
    api_key=API_KEY
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# -----------------------------------------
# Sign Translation
# -----------------------------------------

def interpret_signs(words):

    if not words:

        return "No signs detected."

    signs = " ".join(words)

    prompt = f"""
You are an expert Sign Language Translator.

Convert the detected sign words into a
natural and grammatically correct English sentence.

Detected Signs:
{signs}

Rules:
- Return ONE natural sentence.
- Infer missing grammar when necessary.
- Add connecting words if needed.
- Do not explain.
- Do not return bullet points.
- Return only the final sentence.
"""

    try:

        response = model.generate_content(
            prompt
        )

        return response.text.strip()

    except Exception as e:

        return (
            f"Gemini Error: {str(e)}"
        )
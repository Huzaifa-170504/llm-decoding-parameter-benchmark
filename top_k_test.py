import os
import time
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

QWEN_URL = os.getenv("QWEN_URL")
MODEL_NAME = os.getenv("MODEL_NAME")


# Same prompt used in the main experiment
TEST_PROMPT = (
    "Explain why keyword research is important for SEO. "
    "Give one short paragraph followed by three practical benefits."
)


def generate_response(prompt, temperature, top_k, top_p):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    }

    start_time = time.time()

    response = requests.post(
        QWEN_URL,
        json=payload,
        timeout=300
    )

    end_time = time.time()

    response.raise_for_status()
    result = response.json()

    return result["response"], round(end_time - start_time, 2)


# Top-K-only controlled experiment
top_k_values = [10, 40, 100]

print("=" * 70)
print("CONTROLLED EXPERIMENT: TOP-K")
print("Fixed Temperature = 0.7")
print("Fixed Top-P = 0.90")
print("=" * 70)

for value in top_k_values:

    output, response_time = generate_response(
        TEST_PROMPT,
        temperature=0.7,
        top_k=value,
        top_p=0.90
    )

    print("\n" + "=" * 70)
    print("Temperature: 0.7")
    print("Top-K:", value)
    print("Top-P: 0.90")
    print("Time:", response_time, "seconds")

    print("\nResponse:")
    print(output)

print("\n" + "=" * 70)
print("Top-K experiment completed.")
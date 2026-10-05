"""
| Parameter   | Practical/Valid Range | Ollama Default | Our Experiment    |
|-------------|----------------------:|---------------:|-------------------|
| Temperature | typically 0.0–2.0     | **0.8**        | 0.2, 0.7, 1.1     |
| Top-K       | integer, 1–100        | **40**         | 20, 40, 80        |
| Top-P       | 0.0–1.0               | **0.9**        | 0.70, 0.90, 0.98  |

Note: 
1. These values are well designed for demonstrating the effects. 
2. We don't need to test the absolute extremes because, values such as Temperature 2.0 combined with Top-P 1.0 and very high Top-K could produce unnecessarily unstable output.

"""
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

QWEN_URL = os.getenv("QWEN_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

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

experiments = [
    {"setting": "Conservative", "temperature": 0.2, "top_k": 20, "top_p": 0.70},
    {"setting": "Balanced", "temperature": 0.7, "top_k": 40, "top_p": 0.90},
    {"setting": "Creative", "temperature": 1.1, "top_k": 80, "top_p": 0.98}
]

results = []

for experiment in experiments:
    output, response_time = generate_response(
        TEST_PROMPT,
        experiment["temperature"],
        experiment["top_k"],
        experiment["top_p"]
    )

    results.append({
        "Setting": experiment["setting"],
        "Temperature": experiment["temperature"],
        "Top-K": experiment["top_k"],
        "Top-P": experiment["top_p"],
        "Time (seconds)": response_time,
        "Response": output
    })

    print("\n" + "=" * 70)
    print("Setting:", experiment["setting"])
    print("Temperature:", experiment["temperature"])
    print("Top-K:", experiment["top_k"])
    print("Top-P:", experiment["top_p"])
    print("Time:", response_time, "seconds")
    print("\nResponse:")
    print(output)

df = pd.DataFrame(results)

print("\nExperiment Results:")
print(df[[
    "Setting",
    "Temperature",
    "Top-K",
    "Top-P",
    "Time (seconds)"
]])

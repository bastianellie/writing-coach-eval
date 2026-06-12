"""Quick connectivity test — Azure OpenAI via LiteLLM.

Reads credentials from the .env file, sends a single completion request to
gpt-5o-mini through the configured Azure endpoint, and prints the response.

Usage
-----
    python test_litellm_connection.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
import litellm

load_dotenv()

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------

REQUIRED = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
    print("Make sure they are set in your .env file.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LiteLLM configuration
# ---------------------------------------------------------------------------

litellm.drop_params = True  # silently drop unsupported params

MODEL = "azure/gpt-5-chat"
# API_BASE = os.environ["AZURE_OPENAI_ENDPOINT"]
print(os.environ["AZURE_OPENAI_ENDPOINT"])
print(os.environ["AZURE_OPENAI_API_KEY"])
print(os.environ["AZURE_OPENAI_API_VERSION"])
print(os.environ["AZURE_OPENAI_DEPLOYMENT"])
# API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
# API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
# API_D = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": (
            "In one sentence, explain why writing clearly is important "
            "in academic research."
        ),
    },
]

# ---------------------------------------------------------------------------
# Send request
# ---------------------------------------------------------------------------

# print(f"Sending test completion request to model: {MODEL}")
# print(f"Endpoint : {API_BASE}")
# print(f"API version: {API_VERSION}")
print("-" * 60)

try:
    response = litellm.completion(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=128,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    print("✅  Connection successful!\n")
    print(f"Response:\n  {content}")

except Exception as exc:
    print(f"❌  Request failed: {exc}")
    sys.exit(1)

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


def query_llm(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=20
    )

    data = response.json()

    print("\nRAW RESPONSE:")
    print(data)

    return data.get("response", "")
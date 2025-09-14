import sys
import types
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *a, **k: None
requests_stub.get = lambda *a, **k: None
requests_stub.RequestException = Exception
requests_stub.Response = type("Response", (), {})
sys.modules.setdefault("requests", requests_stub)

import appv2


def test_openrouter_batch_parses_json_with_fences_and_noise():
    client = appv2.OpenRouterClient(api_key="key", model="model", debug=True)

    def fake_post(payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Intro [0]\n```json\n[\"one\", \"two\"]\n```\nOutro {ignored}"
                    }
                }
            ]
        }

    client._post = fake_post
    items = [("customer", "a"), ("agent", "b")]
    assert client.summarise_one_liners_batch(items) == ["one", "two"]


def test_openai_batch_parses_json_with_fences_and_noise():
    client = appv2.OpenAIClient(api_key="key", model="model", debug=True)

    class Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Pre [noise]\n```json\n[\"x\", \"y\"]\n```\nPost {junk}"
                        }
                    }
                ]
            }

    client._post_with_backoff = lambda url, payload, label: Resp()
    items = [("customer", "c"), ("agent", "d")]
    assert client.summarise_one_liners_batch(items) == ["x", "y"]


def test_openrouter_batch_retries_on_truncated_response():
    client = appv2.OpenRouterClient(api_key="key", model="model", debug=True)

    calls = []

    def fake_post(payload):
        user_payload = payload["messages"][1]["content"].split("\n", 1)[1]
        items = json.loads(user_payload)
        calls.append(len(items))
        if len(items) > 1:
            # Truncated JSON: missing closing bracket
            return {"choices": [{"message": {"content": "[\"a\", \"b\""}}]}
        text = items[0]["text"]
        return {"choices": [{"message": {"content": f"[\"{text}-sum\"]"}}]}

    client._post = fake_post
    items = [("customer", "alpha"), ("agent", "beta")]
    assert client.summarise_one_liners_batch(items) == ["alpha-sum", "beta-sum"]
    assert calls == [2, 1, 1]


def test_openai_batch_retries_on_truncated_response():
    client = appv2.OpenAIClient(api_key="key", model="model", debug=True)

    calls = []

    class Resp:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    def fake_post(url, payload, label):
        user_payload = payload["messages"][1]["content"].split("\n", 1)[1]
        items = json.loads(user_payload)
        calls.append(len(items))
        if len(items) > 1:
            return Resp("[\"x\", \"y\"")  # truncated
        text = items[0]["text"]
        return Resp(f"[\"{text}-sum\"]")

    client._post_with_backoff = fake_post
    items = [("customer", "one"), ("agent", "two")]
    assert client.summarise_one_liners_batch(items) == ["one-sum", "two-sum"]
    assert calls == [2, 1, 1]

import sys
import types
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

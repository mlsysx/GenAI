# Model configuration

The starter includes adapters for **Ollama**, **Hugging Face Inference Providers**, and **Gemini**. You may add another provider while preserving `generate(messages, images=..., json_mode=...)`, `Reply`, and `describe()`. The assignment does not require a particular vendor. Use one answer model and identical decoding settings across all three conditions.

Model availability, image support, access and pricing vary. Model IDs are deliberately blank. Choose accessible models, record their exact identifiers, and test both JSON output and actual-image input before a long run. No model is downloaded or contacted on import.

## Configuration fields

| Field | Meaning |
|---|---|
| `answer_backend.provider`, `.model` | Provider and exact available answer model |
| `vision_backend` | Same backend fields for a VLM; `null` reuses the answer model, which must then accept images |
| `temperature`, `max_output_tokens`, `timeout_seconds` | Decoding and per-request bounds |
| `json_mode` | Request provider JSON output where supported; the host still validates every action |
| `manifest` | Path relative to the Homework3 working directory, not the configuration file |
| `embedding_model`, `embedding_revision` | Pretrained model ID/path and revision; record the resolved version |
| `retriever`, `top_k` | `dense` or `lexical`, and 1–6 chunks per retrieval |
| `chunk_max_words`, `chunk_overlap` | Student chunking settings |
| `limits` | Per-episode final-benchmark limits |
| `session_limits`, `max_session_turns` | Cumulative interactive-session limits; persist across turns/restoration |
| `enable_live` | Must be explicitly set to `true` before requests |

Keys stay in `HF_TOKEN` or `GEMINI_API_KEY` environment variables (or a local ignored `.env`). The adapter uses `api_key_env` only as a variable name. Do not put a key in JSON. `python-dotenv` loads `.env` in the notebook and UI. Local Ollama needs no API key.

## Routes

**Ollama:** Install and run Ollama separately, select/download a model you can run, and set provider to `ollama`. The starter uses the loopback endpoint `http://localhost:11434/api/chat`, nonstreaming JSON responses, and base64 image bytes. A separate vision model can be configured. See [chat API](https://docs.ollama.com/api/chat) and [vision](https://docs.ollama.com/capabilities/vision).

**Hugging Face:** Set provider to `huggingface`, an accessible routed model ID, and `HF_TOKEN`. The starter uses `https://router.huggingface.co/v1/chat/completions`. Confirm JSON and image support for your selected provider/model. See [official chat-completion documentation](https://huggingface.co/docs/inference-providers/tasks/chat-completion).

**Gemini:** Set provider to `gemini`, an available model ID, and `GEMINI_API_KEY`. The starter calls `generateContent`, uses inline image bytes, and requests JSON MIME output for agent decisions. See [official API documentation](https://ai.google.dev/api/generate-content).

Hosted routes require `allow_external: true` for every loaded source, based on your actual permission assessment. This flag is an attestation, not an automatic license check. Local sources are loaded from the manifest only.

## Explicit smoke test

After configuring a model and enabling live calls:

```bash
python scripts/check_setup.py --config config.json
python scripts/model_smoke.py --config config.json --live
```

The second command makes one answer call and one pixel-inspection call, saves actual output and usage in a new `outputs/provider_smoke_*` directory, and makes no retries. Inspect its JSON compliance and image observations yourself. This is connectivity/capability evidence, not homework evaluation.

If the selected model lacks JSON mode, set `json_mode: false`; parsing/schema validation remains mandatory. The adapter never silently changes models. Provider/format failures stay in experiment logs. Time limits are checked between calls, so one in-flight request can overshoot the total budget up to its request timeout; choose both limits deliberately.

## Embeddings and reproducibility

Install `requirements_embeddings.txt` only when ready for pretrained embeddings. `build_index(..., allow_model_download=True)` explicitly allows loading/downloading the chosen Sentence Transformer on CPU. See [SentenceTransformer documentation](https://sbert.net/docs/package_reference/sentence_transformer/model.html). Select a model compatible with the supplied `encode` route; if it requires query/document prefixes, implement and document those consistently.

Use `python -m pip freeze > outputs/environment_freeze.txt` for your own environment. No embedding weights or credentials are distributed.


# ComfyUI LMStudio Remote Nodes

ComfyUI custom nodes to connect to a remote LMStudio server through the OpenAI API v1-compatible interface.

## Included nodes

1. **LMStudio - Connect**
   - Server URL (example: `http://10.168.168.7:1234`)
   - API token (`-` placeholder supported)
   - Model dropdown (populated by **Refresh Models** button)
   - **Test Connection** button
   - Reasoning toggle
   - Advanced options (collapsed by default): max tokens, temperature, timeout, tooling/MCP toggle
   - Outputs a reusable `LMSTUDIO_CONNECTION` payload

2. **LMStudio - Text Gen**
   - `Connection` input from LMStudio - Connect
   - System prompt (`LM Souls` style)
   - User prompt
   - Seed (`-1` = secure random)

3. **LMStudio - Image To Text**
   - `Connection` input from LMStudio - Connect
   - Image input
   - System prompt (`LM Souls` style)
   - User prompt
   - Seed (`-1` = secure random)

## Project structure

```
comfyui-lmstudio-node/
├── __init__.py               # Comfy extension entrypoint
├── connect_node.py           # Connection node schema + payload builder
├── text_gen_node.py          # Text generation node
├── image_to_text_node.py     # Image-to-text node
├── client.py                 # OpenAI SDK client + shared helpers
├── models.py                 # Dataclasses for connection payload
├── iotypes.py                # Custom Comfy IO types
├── routes.py                 # Backend routes for model list + connectivity test
├── web/
│   └── lmstudio_connect.js   # Frontend buttons for refresh/test and dropdown update
├── tests/                    # Unit tests
├── requirements.txt
└── pyproject.toml
```

## Behavior and reliability notes

- Uses only the `openai` Python SDK. No legacy HTTP fallback path.
- Seed handling is sanitized and deterministic-safe; no `random.seed()` calls are used.
- Text and image nodes use `responses` endpoint first, then fallback to `chat.completions` if needed.
- MCP/tooling toggle is preserved in connection payload. It is useful only when your LMStudio model/session is configured for tool-capable responses.

## Installation

1. Place this folder under `ComfyUI/custom_nodes/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## LMStudio compatibility references

- [LMStudio OpenAI compatibility docs](https://lmstudio.ai/docs/developer/openai-compat)
- [ComfyUI custom node docs](https://docs.comfy.org/development/core-concepts/custom-nodes)

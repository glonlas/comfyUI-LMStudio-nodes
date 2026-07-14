# ComfyUI LMStudio Nodes (Unofficial)

Small ComfyUI custom nodes to connect to a remote LMStudio server through OpenAI-compatible API (`/v1`).

![LMStudio ComfyUI Nodes](docs/assets/lmstudio-comfyui-screenshot.png)

## Nodes

- **LMStudio - Connect**
  - Connect to a remote LMStudio URL
  - Pick the model from a dropdown populated by **Refresh Models**
  - **Test connectivity** to the server
  - **Thinking / Reasoning** control — `auto` / `on` / `off`:
    - `auto` — defer to the model, but force reasoning on for families that stay
      silent by default (Gemma). Qwen3-style models already think by default.
    - `on` — force reasoning on (`enable_thinking: true`).
    - `off` — suppress reasoning so the model answers directly and doesn't burn
      its token budget on a hidden `<think>` trace.
  - Advanced: API token, Tooling / MCP toggle, max tokens, temperature, timeout

- **LMStudio - Text Gen**
  - System prompt + user prompt
  - Text generation through the shared connection
  - `<think>...</think>` blocks are stripped from response output

- **LMStudio - Image To Text**
  - System prompt + user prompt + image input
  - Image description / caption style generation
  - `<think>...</think>` blocks are stripped from response output

## Quick start

1. Put this folder in `ComfyUI/custom_nodes/`
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI

## Testing

Run the Python unit tests with coverage gate (`>=90%`):

```bash
pytest -q
```

## Notes

- This is **not** an official LMStudio project.
- Not affiliated with LMStudio.
- It is a quickly vibecoded utility project.

## References

- [LMStudio OpenAI compatibility docs](https://lmstudio.ai/docs/developer/openai-compat)
- [ComfyUI custom node docs](https://docs.comfy.org/development/core-concepts/custom-nodes)

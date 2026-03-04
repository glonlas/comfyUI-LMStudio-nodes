from dataclasses import dataclass


@dataclass(frozen=True)
class LMStudioConnectionPayload:
    server_url: str
    base_url: str
    api_key: str
    model: str
    reasoning_enabled: bool
    max_tokens: int
    temperature: float
    timeout_seconds: int
    use_tooling_mcp: bool

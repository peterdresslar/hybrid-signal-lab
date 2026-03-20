from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Prompt:
    id: str
    prompt_text: str
    target: Optional[str] = None
    type: Optional[str] = None
    tokens_approx: Optional[int] = None
    description: Optional[str] = None
    note: Optional[str] = None
    prompt_file: Optional[str] = None
    source: Optional[str] = None

    @classmethod
    def from_text(cls, prompt_text: str, prompt_id: str = "direct_prompt") -> "Prompt":
        return cls(id=prompt_id, prompt_text=prompt_text)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        data_dir: str | Path = "data",
        source: Optional[str] = None,
        default_id: Optional[str] = None,
    ) -> "Prompt":
        prompt_id = payload.get("id") or default_id or "prompt"
        prompt_text = payload.get("prompt")
        prompt_file = payload.get("prompt_file")

        if prompt_text is None and prompt_file:
            prompt_path = Path(prompt_file)
            if not prompt_path.is_file():
                prompt_path = Path(data_dir) / prompt_file
            with open(prompt_path, "r", encoding="utf-8") as file_handle:
                prompt_text = file_handle.read().strip()

        if prompt_text is None:
            raise ValueError(
                "Prompt payload must include either 'prompt' or 'prompt_file'. "
                f"Prompt id: {prompt_id}"
            )

        return cls(
            id=prompt_id,
            prompt_text=prompt_text,
            target=payload.get("target"),
            type=payload.get("type"),
            tokens_approx=payload.get("tokens_approx"),
            description=payload.get("description"),
            note=payload.get("note"),
            prompt_file=prompt_file,
            source=source,
        )

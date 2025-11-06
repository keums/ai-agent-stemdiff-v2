import json
import os
from datetime import datetime
from pathlib import Path

# Default to the memory directory for logs to avoid scattering at repo root
DEFAULT_LOG_PATH = (Path(__file__).parent / "memory_selection_log.jsonl").as_posix()
LOG_PATH = os.getenv("MEMORY_LOG_PATH", DEFAULT_LOG_PATH)


def log_selected_memory(user_prompt: str, file_name: str, reused: bool):
    record = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": user_prompt,
        "selected_file": file_name,
        "reused_latest": reused,
    }

    # Ensure parent directory exists
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

"""Replay hook for deterministically reproducing or resuming prior agent runs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .actions import AddMessages
from .dispatcher import Hook
from .events import LLMQueryEnter, LLMQueryExit


class TrajectoryReplay(Hook):
    """Replays LLM responses from a previous run trajectory."""

    def __init__(self, trajectory_source: Union[str, Path, List[Dict[str, Any]]]):
        super().__init__()
        if isinstance(trajectory_source, (str, Path)):
            with open(trajectory_source, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = list(trajectory_source)
        self.cursor = 0

    def get_next_response(self) -> Optional[str]:
        if self.cursor < len(self.records):
            item = self.records[self.cursor]
            self.cursor += 1
            if isinstance(item, dict):
                return item.get("code") or item.get("response") or item.get("stdout")
            return str(item)
        return None

    def has_more(self) -> bool:
        return self.cursor < len(self.records)

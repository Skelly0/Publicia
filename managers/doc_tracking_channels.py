"""
Manage Google Doc tracking channels stored in a JSON file.
"""
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class DocTrackingChannelManager:
    """Handles persistence of Google Doc tracking channel IDs."""

    def __init__(self, base_dir: str = "documents") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base_dir / "doc_tracking_channels.json"
        self._load()

    def _load(self) -> None:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.channels = [int(x) for x in data if isinstance(x, (int, str)) and str(x).isdigit()]
                    else:
                        logger.warning(f"Invalid format in {self.file_path}, expected list. Resetting.")
                        self.channels = []
            except Exception as e:
                logger.error(f"Error loading {self.file_path}: {e}")
                self.channels = []
        else:
            self.channels = []

    def _save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.channels, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {self.file_path}: {e}")

    def get_channels(self) -> List[int]:
        return list(self.channels)

    def add_channel(self, channel_id: int) -> bool:
        if channel_id not in self.channels:
            self.channels.append(channel_id)
            self._save()
            return True
        return False

    def remove_channel(self, channel_id: int) -> bool:
        if channel_id in self.channels:
            self.channels.remove(channel_id)
            self._save()
            return True
        return False

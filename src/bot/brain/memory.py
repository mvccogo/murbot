import os
import json
import logging
from datetime import datetime
from bot.config import config

logger = logging.getLogger("VoiceBot.Memory")

class MemoryManager:
    """
    Manages persistent memory storage for users and the Discord server.
    Stores facts, preferences, and learned information as JSON files.
    """
    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = storage_dir or config.memories_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.general_file = os.path.join(self.storage_dir, "general_memories.json")
        self._ensure_file(self.general_file)

    def _ensure_file(self, file_path: str):
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _get_user_file(self, user_id: int) -> str:
        return os.path.join(self.storage_dir, f"user_{user_id}.json")

    def save_fact(self, fact: str, category: str = "general", user_id: int | None = None, username: str = "") -> dict:
        """Saves a learned fact to persistent memory."""
        entry = {
            "fact": fact,
            "category": category,
            "username": username,
            "timestamp": datetime.now().isoformat()
        }

        target_file = self._get_user_file(user_id) if user_id else self.general_file
        self._ensure_file(target_file)

        try:
            with open(target_file, "r+", encoding="utf-8") as f:
                try:
                    memories = json.load(f)
                except json.JSONDecodeError:
                    memories = []
                memories.append(entry)
                f.seek(0)
                json.dump(memories, f, ensure_ascii=False, indent=2)
                f.truncate()
            logger.info(f"Saved memory fact: '{fact}' for {username or 'general'}")
            return {"status": "success", "message": f"Fato memorizado com sucesso: '{fact}'"}
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return {"status": "error", "message": str(e)}

    def recall_facts(self, query: str, user_id: int | None = None) -> list[str]:
        """
        Recalls facts relevant to a query from general memory and user memory.
        """
        results = []
        files = [self.general_file]
        if user_id:
            files.append(self._get_user_file(user_id))

        query_terms = [t.lower() for t in query.split() if len(t) > 2]

        for file_path in files:
            if not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    memories = json.load(f)
                    for item in memories:
                        fact_text = item.get("fact", "")
                        # Simple keyword scoring
                        if not query_terms or any(term in fact_text.lower() for term in query_terms):
                            results.append(fact_text)
            except Exception as e:
                logger.error(f"Error reading memory file {file_path}: {e}")

        return results[:10]  # Return top matches

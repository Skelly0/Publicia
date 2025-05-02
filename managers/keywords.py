import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class KeywordManager:
    """Manages loading and querying a keyword database."""

    def __init__(self, db_path: str = "keyword_database.json"):
        """
        Initializes the KeywordManager.

        Args:
            db_path (str): Path to the keyword database JSON file.
        """
        self.db_path = Path(db_path)
        self.keyword_data: Dict[str, str] = {}
        self.keywords: Set[str] = set()
        self._load_database()

    def _load_database(self):
        """Loads the keyword database from the JSON file."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    # Load the database, converting all keys to lowercase
                    raw_data = json.load(f)
                    self.keyword_data = {k.lower(): v for k, v in raw_data.items()}
                    self.keywords = set(self.keyword_data.keys())
                    logger.info(f"Successfully loaded {len(self.keywords)} keywords from {self.db_path}")
            else:
                logger.warning(f"Keyword database file not found at {self.db_path}. Keyword lookup will be disabled.")
                self.keyword_data = {}
                self.keywords = set()
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.db_path}. Keyword lookup disabled.", exc_info=True)
            self.keyword_data = {}
            self.keywords = set()
        except Exception as e:
            logger.error(f"An unexpected error occurred loading keyword database: {e}", exc_info=True)
            self.keyword_data = {}
            self.keywords = set()

    def find_keywords_in_text(self, text: str) -> Set[str]:
        """
        Finds all known keywords present in the given text (case-insensitive).

        Args:
            text (str): The text to search within.

        Returns:
            Set[str]: A set of lowercase keywords found in the text.
        """
        found_keywords = set()
        if not self.keywords or not text:
            return found_keywords

        # Use regex to find whole words matching keywords, case-insensitive
        # \b ensures we match whole words only
        try:
            # Create a regex pattern like \b(keyword1|keyword2|...)\b
            # Escape special regex characters in keywords just in case
            escaped_keywords = [re.escape(kw) for kw in self.keywords]
            pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
            # Find all matches, ignoring case
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Add the lowercase version of each match to the set
            found_keywords.update(match.lower() for match in matches)
        except re.error as e:
            logger.error(f"Regex error while searching for keywords: {e}")
        except Exception as e:
             logger.error(f"Unexpected error finding keywords in text: {e}", exc_info=True)


        return found_keywords

    def get_info_for_keyword(self, keyword: str) -> Optional[str]:
        """
        Retrieves the information associated with a specific keyword.

        Args:
            keyword (str): The keyword (case-insensitive).

        Returns:
            Optional[str]: The information string if the keyword exists, otherwise None.
        """
        return self.keyword_data.get(keyword.lower())

    def reload_database(self):
        """Reloads the keyword database from the file."""
        logger.info(f"Reloading keyword database from {self.db_path}...")
        self._load_database()

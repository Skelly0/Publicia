import json
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, DefaultDict
from collections import defaultdict

# Import Config to check the enabled status
from managers.config import Config

logger = logging.getLogger(__name__)

class KeywordManager:
    """
    Manages loading and querying a keyword database, respecting the enable/disable config.
    Stores multiple definitions per keyword if present in the source file.
    """

    def __init__(self, config: Config, db_path: str = "keyword_database.json"):
        """
        Initializes the KeywordManager.

        Args:
            config (Config): The application configuration object.
            db_path (str): Path to the keyword database JSON file.
        """
        self.config = config
        self.db_path = Path(db_path)
        # Store definitions as a list of strings for each keyword
        self.keyword_data: DefaultDict[str, List[str]] = defaultdict(list)
        self.keywords: Set[str] = set()

        if self.config.KEYWORD_DATABASE_ENABLED:
            self._load_database()
        else:
            logger.info("Keyword database system is disabled via configuration.")

    def _load_database(self):
        """Loads the keyword database from the JSON file if the system is enabled."""
        # Reset data before loading
        self.keyword_data = defaultdict(list)
        self.keywords = set()

        # Double-check enablement in case reload is called directly
        if not self.config.KEYWORD_DATABASE_ENABLED:
             logger.warning("Attempted to load keyword database, but the system is disabled.")
             return # Keep data empty

        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    # Load the database
                    raw_data = json.load(f)
                    count = 0
                    # Iterate and append definitions to the list for each lowercase keyword
                    for k, v in raw_data.items():
                        key_lower = k.lower()
                        self.keyword_data[key_lower].append(v)
                        self.keywords.add(key_lower)
                        count += 1
                    logger.info(f"Successfully loaded {count} definitions for {len(self.keywords)} unique keywords from {self.db_path}")
            else:
                logger.warning(f"Keyword database file not found at {self.db_path}. Keyword lookup will be disabled.")
                # Data remains empty defaultdict
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.db_path}. Keyword lookup disabled.", exc_info=True)
            self.keyword_data = defaultdict(list) # Reset on error
            self.keywords = set()
        except Exception as e:
            logger.error(f"An unexpected error occurred loading keyword database: {e}", exc_info=True)
            self.keyword_data = defaultdict(list) # Reset on error
            self.keywords = set()

    def find_keywords_in_text(self, text: str) -> Set[str]:
        """
        Finds all known keywords present in the given text (case-insensitive).

        Args:
            text (str): The text to search within.

        Returns:
            Set[str]: A set of lowercase keywords found in the text.
        """
        # Return immediately if the system is disabled or no keywords/text
        if not self.config.KEYWORD_DATABASE_ENABLED or not self.keywords or not text:
            return set()

        found_keywords = set()
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

    def get_info_for_keyword(self, keyword: str) -> Optional[List[str]]:
        """
        Retrieves all information entries associated with a specific keyword.

        Args:
            keyword (str): The keyword (case-insensitive).

        Returns:
            Optional[List[str]]: A list of information strings if the keyword exists
                                 and the system is enabled, otherwise None. Returns an empty
                                 list if the keyword exists but has no definitions (should not happen
                                 with current loading logic but handles edge cases).
        """
        if not self.config.KEYWORD_DATABASE_ENABLED:
            return None

        key_lower = keyword.lower()
        # Return the list if the key exists, otherwise None
        return self.keyword_data.get(key_lower)


    def reload_database(self):
        """Reloads the keyword database from the file if the system is enabled."""
        # No need to check enablement here, _load_database does it.
        logger.info(f"Reloading keyword database from {self.db_path}...")
        self._load_database() # _load_database handles the enablement check and resetting data

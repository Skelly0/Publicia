"""
User preferences management for Publicia
"""
import os
import json
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class UserPreferencesManager:
    """Manages user preferences and channel settings."""

    def __init__(self, base_dir: str = "user_preferences"):
        self.preferences_dir = base_dir
        self.channel_settings_file = os.path.join(self.preferences_dir, "channel_settings.json")
        os.makedirs(self.preferences_dir, exist_ok=True)
        self._load_channel_settings() # Load channel settings on init

    def _load_channel_settings(self):
        """Loads channel settings from the JSON file."""
        if os.path.exists(self.channel_settings_file):
            try:
                with open(self.channel_settings_file, 'r', encoding='utf-8') as f:
                    self.channel_settings: Dict[str, Dict[str, Any]] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading channel settings file '{self.channel_settings_file}': {e}. Initializing empty settings.")
                self.channel_settings = {}
        else:
            self.channel_settings = {}

    def _save_channel_settings(self) -> bool:
        """Saves the current channel settings to the JSON file."""
        try:
            with open(self.channel_settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.channel_settings, f, indent=2)
            return True
        except IOError as e:
            logger.error(f"Error saving channel settings file '{self.channel_settings_file}': {e}")
            return False

    def get_file_path(self, user_id: str) -> str:
        """Generate sanitized file path for user preferences."""
        # Ensure user_id is treated as a string for filename safety
        safe_user_id = str(user_id).replace(os.path.sep, '_').replace('..', '_') # Basic sanitization
        return os.path.join(self.preferences_dir, f"{safe_user_id}.json")

    # --- User Specific Preferences ---

    def get_preferred_model(self, user_id: str, default_model: str = None) -> str:
        """Get the user's preferred model, or the default if not set."""
        file_path = self.get_file_path(user_id)

        # Use provided default_model or fallback to "qwen/qwq-32b" if none provided
        if default_model is None:
            default_model = "qwen/qwq-32b"

        if not os.path.exists(file_path):
            return default_model

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("preferred_model", default_model)
        except Exception as e:
            logger.error(f"Error reading user preferences for {user_id}: {e}")
            return default_model

    def set_preferred_model(self, user_id: str, model: str) -> bool:
        """Set the user's preferred model."""
        try:
            file_path = self.get_file_path(user_id)
            preferences = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path} for user {user_id}. Overwriting.")
                    preferences = {} # Reset if invalid

            preferences["preferred_model"] = model

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error setting user preferences for {user_id}: {e}")
            return False

    def get_debug_mode(self, user_id: str) -> bool:
        """Get the user's debug mode preference, default is False."""
        file_path = self.get_file_path(user_id)
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("debug_mode", False)
        except Exception as e:
            logger.error(f"Error reading debug mode preference for {user_id}: {e}")
            return False

    def toggle_debug_mode(self, user_id: str) -> bool:
        """Toggle the user's debug mode preference and return the new state."""
        preferences = {}
        try:
            file_path = self.get_file_path(user_id)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in {file_path} for user {user_id}. Overwriting."
                    )
                    preferences = {}  # Reset if invalid

            current_mode = preferences.get("debug_mode", False)
            new_mode = not current_mode
            preferences["debug_mode"] = new_mode

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return new_mode
        except Exception as e:
            logger.error(f"Error toggling debug mode for {user_id}: {e}")
            # Return the assumed previous state on error? Or a fixed default?
            return preferences.get("debug_mode", False) # Return last known state or default

    def get_informational_prompt_mode(self, user_id: str) -> bool:
        """Get the user's informational prompt mode preference, default is False."""
        file_path = self.get_file_path(user_id)
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("use_informational_prompt", False)
        except Exception as e:
            logger.error(f"Error reading informational prompt mode preference for {user_id}: {e}")
            return False

    def toggle_informational_prompt_mode(self, user_id: str) -> bool:
        """Toggle the user's informational prompt mode preference and return the new state."""
        try:
            file_path = self.get_file_path(user_id)
            preferences = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path} for user {user_id}. Overwriting.")
                    preferences = {} # Reset if invalid

            current_mode = preferences.get("use_informational_prompt", False)
            new_mode = not current_mode
            preferences["use_informational_prompt"] = new_mode

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return new_mode
        except Exception as e:
            logger.error(f"Error toggling informational prompt mode for {user_id}: {e}")
            return preferences.get("use_informational_prompt", False) # Return last known state or default

    def get_pronouns(self, user_id: str) -> str | None:
        """Get the user's pronouns, default is None."""
        file_path = self.get_file_path(user_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("pronouns", None)
        except Exception as e:
            logger.error(f"Error reading pronouns preference for {user_id}: {e}")
            return None

    def set_pronouns(self, user_id: str, pronouns: str) -> bool:
        """Set the user's pronouns."""
        try:
            file_path = self.get_file_path(user_id)
            preferences = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path} for user {user_id}. Overwriting.")
                    preferences = {} # Reset if invalid

            preferences["pronouns"] = pronouns

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error setting pronouns preference for {user_id}: {e}")
            return False

    def get_last_full_context_usage(self, user_id: str) -> str | None:
        """Get the timestamp of the last full context query usage."""
        file_path = self.get_file_path(user_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("last_full_context_usage", None)
        except Exception as e:
            logger.error(f"Error reading last full context usage for {user_id}: {e}")
            return None

    def record_full_context_usage(self, user_id: str) -> bool:
        """Record the current time as the last full context query usage."""
        try:
            file_path = self.get_file_path(user_id)
            preferences = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path} for user {user_id}. Overwriting.")
                    preferences = {} # Reset if invalid

            from datetime import datetime
            preferences["last_full_context_usage"] = datetime.now().isoformat()

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error recording full context usage for {user_id}: {e}")
            return False

    # --- Channel Specific Settings ---

    def get_channel_parsing_settings(self, channel_id: str) -> Tuple[bool, int]:
        """Get the parsing settings for a specific channel. Defaults to (False, 50)."""
        channel_id_str = str(channel_id) # Ensure channel_id is string for dict key
        settings = self.channel_settings.get(channel_id_str, {})
        enabled = settings.get("enabled", False)
        count = settings.get("message_count", 50) # Default to 50 messages if not set
        return (enabled, count)

    def set_channel_parsing_settings(self, channel_id: str, enabled: bool, message_count: int) -> bool:
        """Set the parsing settings for a specific channel."""
        channel_id_str = str(channel_id) # Ensure channel_id is string for dict key
        if channel_id_str not in self.channel_settings:
            self.channel_settings[channel_id_str] = {}

        self.channel_settings[channel_id_str]["enabled"] = enabled
        self.channel_settings[channel_id_str]["message_count"] = message_count

        # Save the updated settings to the file
        return self._save_channel_settings()

    # --- Deprecated User-Specific Channel Parsing (Keep for potential migration/cleanup later if needed) ---
    # These methods are kept temporarily but should not be used by new code.
    # The new channel-specific methods above should be used instead.

    def get_channel_parsing(self, user_id: str) -> Tuple[bool, int]:
        """[DEPRECATED] Get the user's channel parsing preference and message count."""
        logger.warning(f"Deprecated method get_channel_parsing called for user {user_id}. Use get_channel_parsing_settings(channel_id) instead.")
        file_path = self.get_file_path(user_id)
        if not os.path.exists(file_path):
            return (False, 0)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                enabled = preferences.get("channel_parsing_enabled", False)
                count = preferences.get("channel_parsing_count", 0)
                return (enabled, count)
        except Exception as e:
            logger.error(f"Error reading deprecated channel parsing preference for {user_id}: {e}")
            return (False, 0)

    def set_channel_parsing(self, user_id: str, enabled: bool, count: int) -> bool:
        """[DEPRECATED] Set the user's channel parsing preference and message count."""
        logger.warning(f"Deprecated method set_channel_parsing called for user {user_id}. Use set_channel_parsing_settings(channel_id, ...) instead.")
        try:
            file_path = self.get_file_path(user_id)
            preferences = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        preferences = json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path} for user {user_id}. Overwriting.")
                    preferences = {} # Reset if invalid

            preferences["channel_parsing_enabled"] = enabled
            preferences["channel_parsing_count"] = count

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error setting deprecated channel parsing preference for {user_id}: {e}")
            return False

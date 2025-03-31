"""
User preferences management for Publicia
"""
import os
import json
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class UserPreferencesManager:
    """Manages user preferences such as preferred models."""
    
    def __init__(self, base_dir: str = "user_preferences"):
        self.preferences_dir = base_dir
        os.makedirs(self.preferences_dir, exist_ok=True)
    
    def get_file_path(self, user_id: str) -> str:
        """Generate sanitized file path for user preferences."""
        return os.path.join(self.preferences_dir, f"{user_id}.json")
    
    def get_preferred_model(self, user_id: str, default_model: str = None) -> str:
        """Get the user's preferred model, or the default if not set."""
        file_path = self.get_file_path(user_id)
        
        # Use provided default_model or fallback to "qwen/qwq-32b" if none provided
        # This allows the config.DEFAULT_MODEL to be passed in from bot.py
        if default_model is None:
            default_model = "qwen/qwq-32b"
        
        if not os.path.exists(file_path):
            return default_model
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("preferred_model", default_model)
        except Exception as e:
            logger.error(f"Error reading user preferences: {e}")
            return default_model

    def get_channel_parsing(self, user_id: str) -> Tuple[bool, int]:
        """Get the user's channel parsing preference and message count, defaults to (False, 0)."""
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
            logger.error(f"Error reading channel parsing preference: {e}")
            return (False, 0)

    def set_channel_parsing(self, user_id: str, enabled: bool, count: int) -> bool:
        """Set the user's channel parsing preference and message count."""
        try:
            file_path = self.get_file_path(user_id)
            
            # Load existing preferences or create new ones
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        preferences = json.load(file)
                    except json.JSONDecodeError:
                        # If file exists but isn't valid JSON, start fresh
                        preferences = {}
            else:
                preferences = {}
            
            # Update preferences
            preferences["channel_parsing_enabled"] = enabled
            preferences["channel_parsing_count"] = count
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            
            return True
                
        except Exception as e:
            logger.error(f"Error setting channel parsing preference: {e}")
            return False
    
    def set_preferred_model(self, user_id: str, model: str) -> bool:
        """Set the user's preferred model."""
        try:
            file_path = self.get_file_path(user_id)
            
            # Load existing preferences or create new ones
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        preferences = json.load(file)
                    except json.JSONDecodeError:
                        # If file exists but isn't valid JSON, start fresh
                        preferences = {}
            else:
                preferences = {}
            
            # Update preferred model
            preferences["preferred_model"] = model
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            
            return True
                
        except Exception as e:
            logger.error(f"Error setting user preferences: {e}")
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
            logger.error(f"Error reading debug mode preference: {e}")
            return False
    
    def toggle_debug_mode(self, user_id: str) -> bool:
        """Toggle the user's debug mode preference and return the new state."""
        try:
            file_path = self.get_file_path(user_id)
            
            # Load existing preferences or create new ones
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        preferences = json.load(file)
                    except json.JSONDecodeError:
                        preferences = {}
            else:
                preferences = {}
            
            # Toggle debug mode
            current_mode = preferences.get("debug_mode", False)
            preferences["debug_mode"] = not current_mode
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            
            return preferences["debug_mode"]
                
        except Exception as e:
            logger.error(f"Error toggling debug mode: {e}")
            return False

    def get_informational_prompt_mode(self, user_id: str) -> bool:
        """Get the user's informational prompt mode preference, default is False."""
        file_path = self.get_file_path(user_id)
        
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                # Default to False if the key doesn't exist
                return preferences.get("use_informational_prompt", False)
        except Exception as e:
            logger.error(f"Error reading informational prompt mode preference: {e}")
            return False # Default to False on error

    def toggle_informational_prompt_mode(self, user_id: str) -> bool:
        """Toggle the user's informational prompt mode preference and return the new state."""
        try:
            file_path = self.get_file_path(user_id)
            
            # Load existing preferences or create new ones
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        preferences = json.load(file)
                    except json.JSONDecodeError:
                        preferences = {}
            else:
                preferences = {}
            
            # Toggle informational prompt mode, defaulting to False if not present
            current_mode = preferences.get("use_informational_prompt", False)
            preferences["use_informational_prompt"] = not current_mode
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            
            # Return the new state
            return preferences["use_informational_prompt"]
                
        except Exception as e:
            logger.error(f"Error toggling informational prompt mode: {e}")
            # In case of error, we can't be sure of the state, maybe return current state before toggle attempt?
            # For simplicity, returning False, but this could be handled differently.
            return False

    def get_pronouns(self, user_id: str) -> str | None:
        """Get the user's pronouns, default is None."""
        file_path = self.get_file_path(user_id)
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                # Return None if the key doesn't exist
                return preferences.get("pronouns", None)
        except Exception as e:
            logger.error(f"Error reading pronouns preference: {e}")
            return None # Default to None on error

    def set_pronouns(self, user_id: str, pronouns: str) -> bool:
        """Set the user's pronouns."""
        try:
            file_path = self.get_file_path(user_id)
            
            # Load existing preferences or create new ones
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        preferences = json.load(file)
                    except json.JSONDecodeError:
                        preferences = {}
            else:
                preferences = {}
            
            # Update pronouns
            preferences["pronouns"] = pronouns
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(preferences, file, indent=2)
            
            return True
                
        except Exception as e:
            logger.error(f"Error setting pronouns preference: {e}")
            return False

"""
Conversation history management for Publicia
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history for context using JSON format."""
    
    def __init__(self, base_dir: str = "conversations"):
        self.conversation_dir = base_dir
        os.makedirs(self.conversation_dir, exist_ok=True)
        self.migrate_old_conversations()
        self.migrate_old_json_format()
    
    def get_file_path(self, username: str) -> str:
        """Generate sanitized file path for user conversations."""
        sanitized_username = "".join(c for c in username if c.isalnum() or c in (' ', '.', '_')).rstrip()
        return os.path.join(self.conversation_dir, f"{sanitized_username}.json")
    
    def migrate_old_conversations(self):
        """Migrate old text-based conversations to JSON format."""
        try:
            # Find all .txt files in the conversation directory
            txt_files = [f for f in os.listdir(self.conversation_dir) if f.endswith('.txt')]
            migrated_count = 0
            
            for txt_file in txt_files:
                try:
                    username = txt_file[:-4]  # Remove .txt extension
                    old_path = os.path.join(self.conversation_dir, txt_file)
                    new_path = os.path.join(self.conversation_dir, f"{username}.json")
                    
                    # Skip if JSON file already exists
                    if os.path.exists(new_path):
                        continue
                    
                    # Read old conversation
                    with open(old_path, 'r', encoding='utf-8-sig') as f:
                        lines = f.readlines()
                    
                    # Convert to JSON format - as a direct array, not nested in a "messages" object
                    messages = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("User: "):
                            messages.append({
                                "role": "user",
                                "content": line[6:],  # Remove "User: " prefix
                                "timestamp": datetime.now().isoformat(),
                                "channel": "unknown"  # Set default channel
                            })
                        elif line.startswith("Bot: "):
                            messages.append({
                                "role": "assistant",
                                "content": line[5:],  # Remove "Bot: " prefix
                                "timestamp": datetime.now().isoformat(),
                                "channel": "unknown"  # Set default channel
                            })
                    
                    # Write new JSON file
                    with open(new_path, 'w', encoding='utf-8') as f:
                        json.dump(messages, f, indent=2)
                    
                    # Rename old file to .txt.bak
                    os.rename(old_path, f"{old_path}.bak")
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating conversation for {txt_file}: {e}")
            
            logger.info(f"Migrated {migrated_count} conversations to JSON format")
            
        except Exception as e:
            logger.error(f"Error migrating conversations: {e}")

    def migrate_old_json_format(self):
        """Migrate old JSON format with 'messages' key to simpler array format."""
        try:
            # Find all .json files in the conversation directory
            json_files = [f for f in os.listdir(self.conversation_dir) if f.endswith('.json')]
            migrated_count = 0
            
            for json_file in json_files:
                try:
                    file_path = os.path.join(self.conversation_dir, json_file)
                    
                    # Read old JSON format
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing JSON file: {json_file}")
                            continue
                    
                    # Check if it's in the old format (has a 'messages' key)
                    if isinstance(data, dict) and "messages" in data:
                        # Extract messages array
                        messages = data["messages"]
                        
                        # Write new format
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(messages, f, indent=2)
                        
                        migrated_count += 1
                        logger.info(f"Migrated JSON format for {json_file}")
                    
                except Exception as e:
                    logger.error(f"Error migrating JSON format for {json_file}: {e}")
            
            logger.info(f"Migrated {migrated_count} JSON files to new format")
            
        except Exception as e:
            logger.error(f"Error migrating JSON format: {e}")
    
    def read_conversation(self, username: str, limit: int = 10) -> List[Dict]:
        """Read recent conversation messages for a user."""
        file_path = self.get_file_path(username)
        
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                messages = json.load(file)
                # Return the most recent messages up to the limit
                return messages[-limit:]
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON for user {username}")
            return []
        except Exception as e:
            logger.error(f"Error reading conversation: {e}")
            return []
    
    def write_conversation(self, username: str, role: str, content: str, channel: str = None):
        """Append a message to the user's conversation history."""
        try:
            file_path = self.get_file_path(username)
            
            # Load existing conversation or create new one
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        messages = json.load(file)
                    except json.JSONDecodeError:
                        # If file exists but isn't valid JSON, start fresh
                        messages = []
            else:
                messages = []
            
            # Create message object
            message = {
                "role": role,  # "user" or "assistant"
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add channel if provided
            if channel:
                message["channel"] = channel
            
            # Add new message
            messages.append(message)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(messages, file, indent=2)
            
            # Limit conversation size
            self.limit_conversation_size(username)
                
        except Exception as e:
            logger.error(f"Error writing conversation: {e}")
    
    def get_conversation_messages(self, username: str, limit: int = 50) -> List[Dict]:
        """Get conversation history as message objects for LLM."""
        messages = self.read_conversation(username, limit)
        result = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Include channel in content if available and not already present
            channel = msg.get("channel")
            if channel and not content.startswith(f"[{channel}]"):
                content = f"[{channel}] {content}"
            
            result.append({
                "role": msg.get("role", "user"),
                "content": content
            })
        
        return result
    
    def limit_conversation_size(self, username: str, max_messages: int = 50):
        """Limit the conversation to the most recent N messages."""
        try:
            file_path = self.get_file_path(username)
            
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    messages = json.load(file)
                except json.JSONDecodeError:
                    return
            
            # Limit number of messages
            if len(messages) > max_messages:
                messages = messages[-max_messages:]
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(messages, file, indent=2)
                    
        except Exception as e:
            logger.error(f"Error limiting conversation size: {e}")

    def get_limited_history(self, username: str, limit: int = 10) -> List[Dict]:
        """
        Get a limited view of the most recent conversation history with display indices.
        
        Args:
            username: The username
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries with added 'display_index' field
        """
        file_path = self.get_file_path(username)
        
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    messages = json.load(file)
                except json.JSONDecodeError:
                    return []
                    
            # Get the most recent messages
            recent_messages = messages[-limit:]
            
            # Add an index to each message for reference
            for i, msg in enumerate(recent_messages):
                msg['display_index'] = i
                
            return recent_messages
        except Exception as e:
            logger.error(f"Error getting limited history: {e}")
            return []

    def delete_messages_by_display_index(self, username: str, indices: List[int], limit: int = 50) -> Tuple[bool, str, int]:
        """
        Delete messages by their display indices (as shown in get_limited_history).
        
        Args:
            username: The username
            indices: List of display indices to delete
            limit: Maximum number of messages that were displayed
            
        Returns:
            Tuple of (success, message, deleted_count)
        """
        file_path = self.get_file_path(username)
        
        if not os.path.exists(file_path):
            return False, "No conversation history found.", 0
            
        try:
            # Read the current conversation
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    messages = json.load(file)
                except json.JSONDecodeError:
                    return False, "Conversation history appears to be corrupted.", 0
            
            # Check if there are any messages
            if not messages:
                return False, "No messages to delete.", 0
                
            # Calculate the offset for display indices
            offset = max(0, len(messages) - limit)
            
            # Convert display indices to actual indices
            actual_indices = [offset + idx for idx in indices if 0 <= idx < min(limit, len(messages))]
            
            # Check if indices are valid
            if not actual_indices:
                return False, "No valid message indices provided.", 0
                
            if max(actual_indices) >= len(messages) or min(actual_indices) < 0:
                return False, "Invalid message indices after adjustment.", 0
            
            # Remove messages at the specified indices
            # We need to remove from highest index to lowest to avoid shifting issues
            sorted_indices = sorted(actual_indices, reverse=True)
            deleted_count = 0
            
            for idx in sorted_indices:
                messages.pop(idx)
                deleted_count += 1
            
            # Write the updated conversation back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(messages, file, indent=2)
            
            return True, f"Successfully deleted {deleted_count} message(s).", deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting messages: {e}")
            return False, f"Error deleting messages: {str(e)}", 0

    def archive_conversation(self, username: str, archive_name: str = None) -> Tuple[bool, str]:
        """Archive a user's conversation history."""
        try:
            # Get the path to the current conversation file
            current_file_path = self.get_file_path(username)
            
            # Check if the file exists
            if not os.path.exists(current_file_path):
                return False, "No conversation history found to archive."
                
            # Create archives directory if it doesn't exist
            archives_dir = os.path.join(self.conversation_dir, "archives", username)
            os.makedirs(archives_dir, exist_ok=True)
            
            # Generate archive name if not provided
            if not archive_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"archive_{timestamp}"
            else:
                # Sanitize archive name
                archive_name = "".join(c for c in archive_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
            
            # Make sure archive name ends with .json
            if not archive_name.endswith('.json'):
                archive_name += '.json'
            
            # Set the path for the archived file
            archive_file_path = os.path.join(archives_dir, archive_name)
            
            # Copy the current conversation to the archive
            with open(current_file_path, 'r', encoding='utf-8') as current_file:
                conversations = json.load(current_file)
                
                with open(archive_file_path, 'w', encoding='utf-8') as archive_file:
                    json.dump(conversations, archive_file, indent=2)
            
            return True, f"Conversation archived as: {archive_name}"
            
        except Exception as e:
            logger.error(f"Error archiving conversation: {e}")
            return False, f"Error archiving conversation: {str(e)}"

    def list_archives(self, username: str) -> List[str]:
        """List all archived conversations for a user."""
        try:
            # Get the archives directory
            archives_dir = os.path.join(self.conversation_dir, "archives", username)
            
            # Check if the directory exists
            if not os.path.exists(archives_dir):
                return []
                
            # Get all JSON files in the directory
            archives = [f for f in os.listdir(archives_dir) if f.endswith('.json')]
            
            return sorted(archives)
            
        except Exception as e:
            logger.error(f"Error listing archives: {e}")
            return []

    def swap_conversation(self, username: str, archive_name: str) -> Tuple[bool, str]:
        """Swap between current and archived conversations."""
        try:
            # Get the path to the current conversation file
            current_file_path = self.get_file_path(username)
            
            # Get the path to the archived conversation file
            archives_dir = os.path.join(self.conversation_dir, "archives", username)
            archive_file_path = os.path.join(archives_dir, archive_name)
            
            # Make sure archive name ends with .json
            if not archive_name.endswith('.json'):
                archive_file_path += '.json'
            
            # Check if the archived file exists
            if not os.path.exists(archive_file_path):
                return False, f"Archive '{archive_name}' not found."
                
            # Create a temporary archive of the current conversation
            temp_archive_success, temp_archive_message = self.archive_conversation(username, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if not temp_archive_success:
                return False, f"Failed to create temporary archive: {temp_archive_message}"
            
            # Read the archived conversation
            with open(archive_file_path, 'r', encoding='utf-8') as archive_file:
                archived_conversations = json.load(archive_file)
            
            # Write the archived conversation to the current conversation file
            with open(current_file_path, 'w', encoding='utf-8') as current_file:
                json.dump(archived_conversations, current_file, indent=2)
            
            return True, f"Swapped to archive: {archive_name}"
            
        except Exception as e:
            logger.error(f"Error swapping conversation: {e}")
            return False, f"Error swapping conversation: {str(e)}"

    def delete_archive(self, username: str, archive_name: str) -> Tuple[bool, str]:
        """Delete an archived conversation."""
        try:
            # Get the path to the archived conversation file
            archives_dir = os.path.join(self.conversation_dir, "archives", username)
            archive_file_path = os.path.join(archives_dir, archive_name)
            
            # Make sure archive name ends with .json
            if not archive_name.endswith('.json'):
                archive_file_path += '.json'
            
            # Check if the archived file exists
            if not os.path.exists(archive_file_path):
                return False, f"Archive '{archive_name}' not found."
                
            # Delete the file
            os.remove(archive_file_path)
            return True, f"Archive '{archive_name}' deleted."
            
        except Exception as e:
            logger.error(f"Error deleting archive: {e}")
            return False, f"Error deleting archive: {str(e)}"

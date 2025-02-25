from __future__ import annotations
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import torch
import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv
from collections import deque
from textwrap import shorten  # Add this import
import sys
import io
import asyncio
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import discord
from discord import app_commands
from discord.ext import commands
from system_prompt import SYSTEM_PROMPT
from zep_cloud.client import AsyncZep
from zep_cloud import Message as ZepMessage
from typing import Any

# Reconfigure stdout to use UTF-8 with error replacement
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('bot_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def split_message(text, max_length=2000):
        """
        Split a text string into chunks under max_length characters,
        preserving newlines where possible.
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_text = ""
        
        for line in lines:
            if current_chunk:
                test_text = current_text + '\n' + line
            else:
                test_text = line
            
            if len(test_text) > max_length:
                if current_chunk:
                    chunks.append(current_text)
                    current_chunk = [line]
                    current_text = line
                else:
                    # The single line is too long, split it
                    for i in range(0, len(line), max_length):
                        chunks.append(line[i:i+max_length])
                    current_chunk = []
                    current_text = ""
            else:
                current_chunk.append(line)
                current_text = test_text
        
        if current_chunk:
            chunks.append(current_text)
        
        return chunks
        
def sanitize_for_logging(text: str) -> str:
    """Remove problematic characters like BOM from the string for safe logging."""
    return text.replace('\ufeff', '')

class DocumentManager:
    """Manages document storage, embeddings, and retrieval."""
    
    def __init__(self, base_dir: str = "documents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        
        # Storage for documents and embeddings
        self.chunks: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Load existing documents
        self._load_documents()
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def add_document(self, name: str, content: str):
        """Add a new document to the system."""
        try:
            # Create chunks
            chunks = self._chunk_text(content)
            
            # Generate embeddings
            embeddings = self.model.encode(chunks)
            
            # Store document data
            self.chunks[name] = chunks
            self.embeddings[name] = embeddings
            self.metadata[name] = {
                'added': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }
            
            # Save to disk
            self._save_to_disk()
            
            logger.info(f"Added document: {name} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document {name}: {e}")
            raise
            
    def get_googledoc_id_mapping(self):
        """get mapping from document names to google doc IDs."""
        tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
        if not tracked_file.exists():
            return {}
        
        with open(tracked_file, 'r') as f:
            tracked_docs = json.load(f)
        
        # create a mapping from document names to doc IDs
        mapping = {}
        for doc in tracked_docs:
            doc_id = doc['id']
            name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
            if not name.endswith('.txt'):
                name += '.txt'
            mapping[name] = doc_id
        
        return mapping
    
    # Update the DocumentManager.search method
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for relevant document chunks."""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            results = []
            logger.info(f"Searching documents for query: {shorten(query, width=100, placeholder='...')}")
            
            # Search each document
            for doc_name, doc_embeddings in self.embeddings.items():
                # Calculate similarities
                similarities = np.dot(doc_embeddings, query_embedding) / (
                    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top chunks
                top_indices = np.argsort(similarities)[-top_k:]
                
                for idx in top_indices:
                    results.append((
                        doc_name,
                        self.chunks[doc_name][idx],
                        float(similarities[idx])
                    ))
            
            # Sort by similarity
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Log search results
            for doc_name, chunk, similarity in results[:top_k]:
                logger.info(f"Found relevant chunk in {doc_name} (similarity: {similarity:.2f})")
                logger.info(f"Chunk content: {shorten(sanitize_for_logging(chunk), width=300, placeholder='...')}")
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _save_to_disk(self):
        """Save document data to disk."""
        try:
            # Save chunks
            with open(self.base_dir / 'chunks.pkl', 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save embeddings
            with open(self.base_dir / 'embeddings.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with open(self.base_dir / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
            raise
    
    def _load_documents(self):
        """Load document data from disk and add any new .txt files."""
        try:
            # Load existing processed data if it exists
            if (self.base_dir / 'chunks.pkl').exists():
                with open(self.base_dir / 'chunks.pkl', 'rb') as f:
                    self.chunks = pickle.load(f)
                with open(self.base_dir / 'embeddings.pkl', 'rb') as f:
                    self.embeddings = pickle.load(f)
                with open(self.base_dir / 'metadata.json', 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} documents from processed data")
            else:
                self.chunks = {}
                self.embeddings = {}
                self.metadata = {}
                logger.info("No processed data found, starting fresh")

            # Find .txt files that are not already loaded
            existing_names = set(self.chunks.keys())
            txt_files = [f for f in self.base_dir.glob('*.txt') if f.name not in existing_names]
            
            if txt_files:
                logger.info(f"Found {len(txt_files)} new .txt files to load")
                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                        self.add_document(txt_file.name, content)
                        logger.info(f"Loaded and processed {txt_file.name}")
                    except Exception as e:
                        logger.error(f"Error processing {txt_file.name}: {e}")
            else:
                logger.info("No new .txt files to load")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self.chunks = {}
            self.embeddings = {}
            self.metadata = {}
            
    def get_lorebooks_path(self):
        """Get or create lorebooks directory path."""
        base_path = Path(self.base_dir).parent / "lorebooks"
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path
    
    def track_google_doc(self, doc_id, name=None):
        """Add a Google Doc to tracked list."""
        # Load existing tracked docs
        tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
        if tracked_file.exists():
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
        else:
            tracked_docs = []
        
        
        # Add new doc if not already tracked
        for doc in tracked_docs:
            if doc['id'] == doc_id:
                return f"Google Doc {doc_id} already tracked"
        
        tracked_docs.append({
            'id': doc_id,
            'custom_name': name,
            'added_at': datetime.now().isoformat()
        })
        
        # Save updated list
        with open(tracked_file, 'w') as f:
            json.dump(tracked_docs, f)
        
        return f"Added Google Doc {doc_id} to tracked list"

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.LLM_MODEL = os.getenv('LLM_MODEL')
        self.ZEP_API_KEY = os.getenv('ZEP_API_KEY')  # Add Zep API key
        
        # Validate required environment variables
        self._validate_config()
        
        # Add timeout settings
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    
    def _validate_config(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            'DISCORD_BOT_TOKEN',
            'OPENROUTER_API_KEY',
            'LLM_MODEL',
            'ZEP_API_KEY'  # Add Zep API key to required vars
        ]
        
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
class ZepMemoryManager:
    """Manages Zep memory integration for long-term knowledge."""
    
    def __init__(self, api_key: str):
        self.zep_client = AsyncZep(api_key=api_key)
        self.user_session_mapping = {}  # Maps Discord usernames to Zep user IDs and session IDs
        
    async def ensure_user_exists(self, username: str, user_id: Optional[str] = None) -> str:
        """Ensure a user exists in Zep, creating one if needed."""
        if username in self.user_session_mapping and 'user_id' in self.user_session_mapping[username]:
            return self.user_session_mapping[username]['user_id']
            
        # If user_id not provided, generate one based on username
        if not user_id:
            import uuid
            user_id = f"{username}_{uuid.uuid4().hex[:8]}"
            
        try:
            # Add user to Zep
            await self.zep_client.user.add(
                user_id=user_id,
                email=f"{username}",  # Fictional email format
                first_name=username,
                last_name=""  # Generic last name
            )
            
            # Store user_id in mapping
            if username not in self.user_session_mapping:
                self.user_session_mapping[username] = {}
                
            self.user_session_mapping[username]['user_id'] = user_id
            return user_id
        except Exception as e:
            logging.error(f"Error ensuring Zep user exists: {e}")
            # Return user_id anyway as the error might be that user already exists
            return user_id
            
    async def ensure_session_exists(self, username: str) -> str:
        """Ensure a session exists for the user, creating one if needed."""
        if username in self.user_session_mapping and 'session_id' in self.user_session_mapping[username]:
            return self.user_session_mapping[username]['session_id']
            
        # Get user_id first
        user_id = await self.ensure_user_exists(username)
        
        # Create new session
        import uuid
        session_id = uuid.uuid4().hex
        
        try:
            await self.zep_client.memory.add_session(
                user_id=user_id,
                session_id=session_id,
            )
            
            # Store in mapping
            self.user_session_mapping[username]['session_id'] = session_id
            return session_id
        except Exception as e:
            logging.error(f"Error creating Zep session: {e}")
            return session_id
            
    async def add_message(self, username: str, role: str, role_type: str, content: str, channel: str = None):
        """Add a message to Zep memory."""
        try:
            session_id = await self.ensure_session_exists(username)
            
            message = ZepMessage(
                role=role,
                role_type=role_type,
                content=content
            )
            
            await self.zep_client.memory.add(
                session_id=session_id,
                messages=[message]
            )
            
            return True
        except Exception as e:
            logging.error(f"Error adding message to Zep: {e}")
            return False
            
    async def find_facts_to_remove(self, username: str, search_text: str, limit: int = 5):
        """
        Find facts that match a search query, including their UUIDs for removal.
        """
        try:
            user_id = await self.ensure_user_exists(username)
            
            # Search using Zep's graph search
            results = await self.zep_client.graph.search(
                user_id=user_id,
                query=search_text,
                limit=limit,
                scope="edges"  # Search for facts (edges)
            )
            
            # Format the results
            facts = []
            for edge in results.edges:
                if hasattr(edge, 'fact') and edge.fact and hasattr(edge, 'uuid_'):
                    facts.append({
                        "fact": edge.fact,
                        "uuid": edge.uuid_
                    })
                        
            return facts
        except Exception as e:
            logging.error(f"Error finding facts to remove: {e}")
            return []

    async def delete_fact_by_uuid(self, uuid: str):
        """Delete a fact by its UUID."""
        try:
            # Using the edge subgroup of the graph API
            await self.zep_client.graph.edge.delete(uuid=uuid)
            return True
        except Exception as e:
            logging.error(f"Error deleting fact: {e}")
            return False

    async def remove_fact(self, username: str, search_text: str) -> dict:
        """
        Remove a fact from the user's knowledge graph based on a search.
        """
        try:
            # Search for facts matching the search text, including UUIDs
            matched_facts = await self.find_facts_to_remove(username, search_text)
            
            # If no facts match, return early
            if not matched_facts:
                return {
                    "success": False,
                    "message": "No facts found matching your search.",
                    "matched_count": 0,
                    "matched_facts": []
                }
            
            # If multiple facts match, return them and ask for more specific query
            if len(matched_facts) > 1:
                return {
                    "success": False,
                    "message": f"Found {len(matched_facts)} matching facts. Please be more specific.",
                    "matched_count": len(matched_facts),
                    "matched_facts": [fact["fact"] for fact in matched_facts]
                }
            
            # If we get here, we have exactly one matching fact
            fact_to_remove = matched_facts[0]
            
            # Delete the fact
            success = await self.delete_fact_by_uuid(fact_to_remove["uuid"])
            
            if success:
                return {
                    "success": True,
                    "message": "Successfully removed the fact.",
                    "matched_count": 1,
                    "matched_facts": [fact_to_remove["fact"]]
                }
            else:
                return {
                    "success": False,
                    "message": "Error removing the fact.",
                    "matched_count": 1,
                    "matched_facts": [fact_to_remove["fact"]]
                }
            
        except Exception as e:
            logging.error(f"Error removing fact: {e}")
            return {
                "success": False,
                "message": f"Error removing fact: {str(e)}",
                "matched_count": 0,
                "matched_facts": []
            }
            
    async def get_memory_context(self, username: str) -> str:
        """Get memory context for the user."""
        try:
            session_id = await self.ensure_session_exists(username)
            
            # Get memory from Zep
            memory = await self.zep_client.memory.get(session_id=session_id)
            
            # Return the context
            return memory.context
        except Exception as e:
            logging.error(f"Error getting Zep memory: {e}")
            return ""
            
    async def search_memory(self, username: str, query: str, limit: int = 5):
        """Search user's memory for relevant information."""
        try:
            user_id = await self.ensure_user_exists(username)
            
            # Search using Zep's graph search
            results = await self.zep_client.graph.search(
                user_id=user_id,
                query=query,
                limit=limit,
                scope="edges"  # Search for facts (edges)
            )
            
            # Format the results
            facts = []
            for edge in results.edges:
                if hasattr(edge, 'fact') and edge.fact:
                    facts.append(edge.fact)
                    
            return facts
        except Exception as e:
            logging.error(f"Error searching Zep memory: {e}")
            return []
            
    async def add_fact(self, username: str, fact: str):
        """Add a standalone fact to the user's knowledge graph."""
        try:
            user_id = await self.ensure_user_exists(username)
            
            # Use the graph.add method to add unstructured text as a fact
            await self.zep_client.graph.add(
                user_id=user_id,
                data=fact,
                type="text"  # We're adding plain text
            )
            
            return True
        except Exception as e:
            logging.error(f"Error adding fact to Zep: {e}")
            return False


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
    
    def get_conversation_messages(self, username: str, limit: int = 10) -> List[Dict]:
        """Get conversation history as message objects for LLM."""
        messages = self.read_conversation(username, limit)
        result = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Include channel in content if available
            channel = msg.get("channel")
            if channel:
                content = f"[Channel: {channel}] {content}"
            
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

class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="Publicia! ", intents=intents)

        self.config = Config()
        self.conversation_manager = ConversationManager()
        self.document_manager = DocumentManager()
        self.timeout_duration = 30
        # Initialize Zep memory manager
        self.zep_memory = ZepMemoryManager(api_key=self.config.ZEP_API_KEY)

        self.banned_users = set()
        self.banned_users_file = 'banned_users.json'
        self.load_banned_users()

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.refresh_google_docs, 'interval', hours=6)
        self.scheduler.start()

    def load_banned_users(self):
        """Load banned users from JSON file."""
        try:
            with open(self.banned_users_file, 'r') as f:
                data = json.load(f)
                self.banned_users = set(data.get('banned_users', []))
        except FileNotFoundError:
            self.banned_users = set()

    def save_banned_users(self):
        """Save banned users to JSON file."""
        with open(self.banned_users_file, 'w') as f:
            json.dump({'banned_users': list(self.banned_users)}, f)

    async def send_split_message(self, channel, text, reference=None, mention_author=False):
        """Helper method to send messages, splitting them if they exceed 2000 characters."""
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            await channel.send(
                chunk,
                reference=reference if i == 0 else None,
                mention_author=mention_author if i == 0 else False
            )

    async def refresh_google_docs(self):
        """Refresh all tracked Google Docs."""
        tracked_file = Path(self.document_manager.base_dir) / "tracked_google_docs.json"
        if not tracked_file.exists():
            return
            
        with open(tracked_file, 'r') as f:
            tracked_docs = json.load(f)
        
        for doc in tracked_docs:
            try:
                doc_id = doc['id']
                file_name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                if not file_name.endswith('.txt'):
                    file_name += '.txt'
                
                async with aiohttp.ClientSession() as session:
                    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.error(f"Failed to download {doc_id}: {response.status}")
                            continue
                        content = await response.text()
                
                file_path = self.document_manager.base_dir / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info(f"Updated Google Doc {doc_id} as {file_name}")
                
                if file_name in self.document_manager.chunks:
                    del self.document_manager.chunks[file_name]
                    del self.document_manager.embeddings[file_name]
                    
                self.document_manager.add_document(file_name, content)
                
            except Exception as e:
                logger.error(f"Error refreshing doc {doc_id}: {e}")

    async def setup_hook(self):
        """Initial setup hook called by discord.py."""
        logger.info("Bot is setting up...")
        await self.setup_commands()
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def setup_commands(self):
        """Set up all slash commands."""
        # Existing commands setup code...
        
        # Add new commands for Zep memory management
        @self.tree.command(name="add_fact", description="Add a fact to the bot's long-term memory")
        @app_commands.describe(
            fact="The fact to remember"
        )
        async def add_fact(interaction: discord.Interaction, fact: str):
            await interaction.response.defer()
            try:
                success = await self.zep_memory.add_fact(interaction.user.name, fact)
                if success:
                    await interaction.followup.send(f"*neurons activated!* fact added to my long-term memory: \"{fact}\"")
                else:
                    await interaction.followup.send("*my brain misfired...* couldn't add fact to long-term memory.")
            except Exception as e:
                logger.error(f"Error adding fact: {e}")
                await interaction.followup.send("*neural pathways disrupted!* couldn't process fact.")
                
        @self.tree.command(name="remove_fact", description="Remove a fact from the bot's long-term memory")
        @app_commands.describe(
            query="Search text to find the fact to remove"
        )
        async def remove_fact(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                result = await self.zep_memory.remove_fact(interaction.user.name, query)
                
                if result["success"]:
                    await interaction.followup.send(f"*neural circuit pruned!* removed fact: \"{result['matched_facts'][0]}\"")
                else:
                    if result["matched_count"] == 0:
                        await interaction.followup.send(f"*memory scan complete...* no facts found matching \"{query}\"")
                    else:
                        response = f"*multiple matching facts found ({result['matched_count']})... please be more specific:*\n```"
                        for i, fact in enumerate(result["matched_facts"], 1):
                            response += f"\n{i}. {fact}"
                        response += "```"
                        await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Error removing fact: {e}")
                await interaction.followup.send("*neural pathways disrupted!* couldn't remove fact.")
                
        @self.tree.command(name="search_memory", description="Search the bot's long-term memory")
        @app_commands.describe(
            query="What to search for in memory"
        )
        async def search_memory(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                facts = await self.zep_memory.search_memory(interaction.user.name, query)
                if facts:
                    response = "*accessing neural archives...*\n\nRelevant memories:\n```"
                    for i, fact in enumerate(facts, 1):
                        response += f"\n{i}. {fact}"
                    response += "```"
                else:
                    response = "*my enhanced cortex contains no relevant memories...*"
                    
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                await interaction.followup.send("*synaptic failure...* couldn't search memory circuits.")

        @self.tree.command(name="listcommands", description="List all available commands")
        async def list_commands(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                response = "*accessing command database through enhanced synapses...*\n\n"
                response += "**AVAILABLE COMMANDS**\n\n"
                
                categories = {
                    "Lore Queries": ["query"],
                    "Document Management": ["adddoc", "listdocs", "removedoc", "searchdocs", "add_googledoc", "list_googledocs", "remove_googledoc"],
                    "Utility": ["listcommands"],
                    "Memory Management": ["lobotomise", "add_fact", "search_memory", "remove_fact"],  # Added remove_fact
                    "Moderation": ["ban_user", "unban_user"]
                }
                
                for category, cmd_list in categories.items():
                    response += f"__*{category}*__\n"
                    for cmd_name in cmd_list:
                        cmd = self.tree.get_command(cmd_name)
                        if cmd:
                            desc = cmd.description or "No description available"
                            response += f"`/{cmd_name}`: {desc}\n"
                    response += "\n"
                
                response += "\n*you can ask questions about ledus banum 77 and imperial lore by mentioning me or using the /query command!*"
                response += "\n*you can also type \"LOBOTOMISE\" in a message to wipe your conversation history.*"
                response += "\n*use /add_fact to teach me important information, /search_memory to find what i've learned, and /remove_fact to prune incorrect memories!*"
                response += "\n\n*my genetically enhanced brain is always ready to help... just ask!*"
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            except Exception as e:
                logger.error(f"Error listing commands: {e}")
                await interaction.followup.send("*my enhanced neurons misfired!* couldn't retrieve command list right now...")
                
        @self.tree.command(name="lobotomise", description="Wipe your conversation history with the bot")
        async def lobotomise(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                file_path = self.conversation_manager.get_file_path(interaction.user.name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    await interaction.followup.send("*AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA*... memory wiped! I've forgotten our conversations... Who are you again?")
                else:
                    await interaction.followup.send("hmm, i don't seem to have any memories of our conversations to wipe!")
            except Exception as e:
                logger.error(f"Error clearing conversation history: {e}")
                await interaction.followup.send("oops, something went wrong while trying to clear my memory!")

        @self.tree.command(name="listdocs", description="List all available documents")
        async def list_documents(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                if not self.document_manager.metadata:
                    await interaction.followup.send("No documents found in the knowledge base.")
                    return
                    
                response = "Available documents:\n```"
                for doc_name, meta in self.document_manager.metadata.items():
                    chunks = meta['chunk_count']
                    added = meta['added']
                    response += f"\n{doc_name} - {chunks} chunks (Added: {added})"
                response += "```"
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            except Exception as e:
                await interaction.followup.send(f"Error listing documents: {str(e)}")
                
                
        @self.tree.command(name="query", description="Ask Publicia a question about Ledus Banum 77 and Imperial lore")
        @app_commands.describe(question="Your question about the lore")
        async def query_lore(interaction: discord.Interaction, question: str):
            await interaction.response.defer()
            try:
                # Get channel name for context
                if interaction.guild:
                    channel_name = interaction.channel.name
                else:
                    channel_name = "DM"
                    
                # Get nickname or username
                nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name
                
                # Get conversation history for context
                conversation_messages = self.conversation_manager.get_conversation_messages(interaction.user.name)
                
                # Retrieve memory context from Zep
                memory_context = await self.zep_memory.get_memory_context(interaction.user.name)
                
                # Add the question to Zep memory
                await self.zep_memory.add_message(
                    username=interaction.user.name,
                    role=nickname,
                    role_type="user",
                    content=question,
                    channel=channel_name
                )
                
                # Load Google Doc ID mapping
                googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

                # Search documents for relevant context
                relevant_docs = self.document_manager.search(question)

                # Import for URL encoding
                import urllib.parse

                # Format context with citation info
                doc_contexts = []
                for doc, chunk, sim in relevant_docs:
                    if doc in googledoc_mapping:
                        # This is a Google Doc, create search URL
                        doc_id = googledoc_mapping[doc]
                        
                        # Get first 10 words as search text
                        words = chunk.split()
                        search_text = ' '.join(words[:min(10, len(words))])
                        
                        # URL encode the search text
                        encoded_search = urllib.parse.quote(search_text)
                        
                        # Create the URL with findtext parameter
                        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                        
                        doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {sim:.2f}):\n{chunk}")
                    else:
                        # Regular document, no citation info
                        doc_contexts.append(f"From document '{doc}' (similarity: {sim:.2f}):\n{chunk}")

                doc_context = "\n\n".join(doc_contexts)
                
                # Prepare messages for AI, now including Zep memory context
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *conversation_messages,
                    {
                        "role": "system",
                        "content": f"Document context that may or may not be relevant:\n{doc_context}"
                    },
                    {
                        "role": "system",
                        "content": f"Long-term memory context about this user and previous interactions:\n{memory_context}"
                    },
                    {
                        "role": "user",
                        "content": f"You are responding to a message in the Discord channel: {channel_name}"
                    },
                    {
                        "role": "user",
                        "content": f"{nickname}: {question}"
                    }
                ]
                
                # message
                
                # Get AI response
                completion = await self._try_ai_completion(
                    self.config.LLM_MODEL,
                    messages,
                    temperature=0.1
                )

                if completion and completion.get('choices'):
                    response = completion['choices'][0]['message']['content']
                    
                    # Update conversation history
                    self.conversation_manager.write_conversation(
                        interaction.user.name,
                        "user",
                        question,
                        channel_name
                    )
                    self.conversation_manager.write_conversation(
                        interaction.user.name,
                        "assistant",
                        response,
                        channel_name
                    )
                    
                    # Add assistant response to Zep memory
                    await self.zep_memory.add_message(
                        username=interaction.user.name,
                        role="Publicia",
                        role_type="assistant",
                        content=response,
                        channel=channel_name
                    )

                    # Send the response, splitting if necessary
                    chunks = split_message(response)
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send("I apologize, but I'm having trouble generating a response right now.")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                await interaction.followup.send("An error occurred while processing your query.")

        @self.tree.command(name="searchdocs", description="Search the document knowledge base")
        @app_commands.describe(query="What to search for")
        async def search_documents(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                results = self.document_manager.search(query, top_k=3)
                if not results:
                    await interaction.followup.send("No relevant documents found.")
                    return
                response = "Search results:\n```"
                for doc_name, chunk, similarity in results:
                    response += f"\nFrom {doc_name} (similarity: {similarity:.2f}):\n"
                    response += f"{chunk[:200]}...\n"
                response += "```"
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            except Exception as e:
                await interaction.followup.send(f"Error searching documents: {str(e)}")

        @self.tree.command(name="removedoc", description="Remove a document from the knowledge base")
        @app_commands.describe(name="Name of the document to remove")
        async def remove_document(interaction: discord.Interaction, name: str):
            await interaction.response.defer()
            try:
                if name in self.document_manager.metadata:
                    del self.document_manager.chunks[name]
                    del self.document_manager.embeddings[name]
                    del self.document_manager.metadata[name]
                    self.document_manager._save_to_disk()
                    await interaction.followup.send(f"Removed document: {name}")
                else:
                    await interaction.followup.send(f"Document not found: {name}")
            except Exception as e:
                await interaction.followup.send(f"Error removing document: {str(e)}")

        @self.tree.command(name="add_googledoc", description="Add a Google Doc to the tracked list")
        @app_commands.describe(
            doc_url="Google Doc URL or ID",
            name="Custom name for the document (optional)"
        )
        async def add_google_doc(interaction: discord.Interaction, doc_url: str, name: str = None):
            await interaction.response.defer()
            try:
                # Extract Google Doc ID from URL if a URL is provided
                if "docs.google.com" in doc_url:
                    # Extract the ID from various Google Docs URL formats
                    if "/d/" in doc_url:
                        doc_id = doc_url.split("/d/")[1].split("/")[0].split("?")[0]
                    elif "id=" in doc_url:
                        doc_id = doc_url.split("id=")[1].split("&")[0]
                    else:
                        await interaction.followup.send("*could not extract doc id from url... is this a valid google docs link?*")
                        return
                else:
                    # Assume the input is already a Doc ID
                    doc_id = doc_url
                
                result = self.document_manager.track_google_doc(doc_id, name)
                await interaction.followup.send(f"*synapses connecting to document ({doc_url})*\n{result}")
                await interaction.followup.send("*initiating neural download sequence...*")
                await self.refresh_google_docs()
                await interaction.followup.send("*neural pathways successfully connected!*")
            except Exception as e:
                logger.error(f"Error adding Google Doc: {e}")
                await interaction.followup.send(f"*my enhanced brain had a glitch!* couldn't add document: {str(e)}")

        @self.tree.command(name="list_googledocs", description="List all tracked Google Docs")
        async def list_google_docs(interaction: discord.Interaction):
            await interaction.response.defer()
            tracked_file = Path(self.document_manager.base_dir) / "tracked_google_docs.json"
            if not tracked_file.exists():
                await interaction.followup.send("*no google docs detected in my neural network...*")
                return
                
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
            
            if not tracked_docs:
                await interaction.followup.send("*my neural pathways show no connected google docs*")
                return
                
            response = "*accessing neural connections to google docs...*\n\n**TRACKED DOCUMENTS**\n```"
            for doc in tracked_docs:
                doc_id = doc['id']
                name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                doc_url = f"https://docs.google.com/document/d/{doc_id}"
                response += f"\n{name} - URL: {doc_url}"
            response += "```"
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
                
        @self.tree.command(name="remove_googledoc", description="Remove a Google Doc from the tracked list")
        @app_commands.describe(
            identifier="Google Doc ID, URL, or custom name to remove"
        )
        async def remove_google_doc(interaction: discord.Interaction, identifier: str):
            await interaction.response.defer()
            try:
                # Path to the tracked docs file
                tracked_file = Path(self.document_manager.base_dir) / "tracked_google_docs.json"
                if not tracked_file.exists():
                    await interaction.followup.send("*no tracked google docs found in my memory banks!*")
                    return
                    
                # Load existing tracked docs
                with open(tracked_file, 'r') as f:
                    tracked_docs = json.load(f)
                
                # Extract Google Doc ID from URL if a URL is provided
                extracted_id = None
                if "docs.google.com" in identifier:
                    # Extract the ID from various Google Docs URL formats
                    if "/d/" in identifier:
                        extracted_id = identifier.split("/d/")[1].split("/")[0].split("?")[0]
                    elif "id=" in identifier:
                        extracted_id = identifier.split("id=")[1].split("&")[0]
                
                # Try to find and remove the doc
                removed = False
                for i, doc in enumerate(tracked_docs):
                    # Priority: 1. Direct ID match, 2. Custom name match, 3. Extracted URL ID match
                    if doc['id'] == identifier or \
                       (doc.get('custom_name') and doc.get('custom_name') == identifier) or \
                       (extracted_id and doc['id'] == extracted_id):
                        removed_doc = tracked_docs.pop(i)
                        removed = True
                        break
                
                if not removed:
                    await interaction.followup.send(f"*hmm, i couldn't find a document matching '{identifier}' in my neural network*")
                    return
                
                # Save updated list
                with open(tracked_file, 'w') as f:
                    json.dump(tracked_docs, f)
                    
                # Get the original URL and document name for feedback
                doc_id = removed_doc['id']
                doc_url = f"https://docs.google.com/document/d/{doc_id}"
                doc_name = removed_doc.get('custom_name') or f"googledoc_{doc_id}"
                
                await interaction.followup.send(f"*i've surgically removed the neural connection to {doc_name}*\n*url: {doc_url}*\n\n*note: document content might still be in my memory. use `/listdocs` to check and `/removedoc` if needed*")
                
            except Exception as e:
                logger.error(f"Error removing Google Doc: {e}")
                await interaction.followup.send(f"*my enhanced brain experienced an error!* couldn't remove document: {str(e)}")

        async def check_permissions(interaction: discord.Interaction):
            if not interaction.guild:
                raise app_commands.CheckFailure("This command can only be used in a server")
            member = interaction.guild.get_member(interaction.user.id)
            return (member.guild_permissions.administrator or 
                    interaction.user.id == 203229662967627777)

        @self.tree.command(name="ban_user", description="Ban a user from using the bot (admin only)")
        @app_commands.describe(user="User to ban")
        @app_commands.check(check_permissions)
        async def ban_user(interaction: discord.Interaction, user: discord.User):
            await interaction.response.defer()
            if user.id in self.banned_users:
                await interaction.followup.send(f"{user.name} is already banned.")
            else:
                self.banned_users.add(user.id)
                self.save_banned_users()
                await interaction.followup.send(f"Banned {user.name} from using the bot.")
                logger.info(f"User {user.name} (ID: {user.id}) banned by {interaction.user.name}")

        @self.tree.command(name="unban_user", description="Unban a user (admin only)")
        @app_commands.describe(user="User to unban")
        @app_commands.check(check_permissions)
        async def unban_user(interaction: discord.Interaction, user: discord.User):
            await interaction.response.defer()
            if user.id not in self.banned_users:
                await interaction.followup.send(f"{user.name} is not banned.")
            else:
                self.banned_users.remove(user.id)
                self.save_banned_users()
                await interaction.followup.send(f"Unbanned {user.name}.")
                logger.info(f"User {user.name} (ID: {user.id}) unbanned by {interaction.user.name}")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        
        
    async def _try_ai_completion(self, model: str, messages: List[Dict], **kwargs) -> Optional[any]:
        """Get AI completion with fallback options."""
        
        models = [
            model,  # Try the requested model first
            "deepseek/deepseek-r1:floor",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1:nitro",
            "deepseek/deepseek-chat",
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-r1-distill-llama-70b",
            "deepseek/deepseek-r1-distill-qwen-32b"
        ]
        
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "Publicia - DPS",
            "Content-Type": "application/json"
        }

        for current_model in models:
            try:
                logger.info(f"Attempting completion with model: {current_model}")
                
                payload = {
                    "model": current_model,
                    "messages": messages,
                    **kwargs
                }
                
                # Log the sanitized messages (removing potential sensitive info)
                sanitized_messages = [
                    {
                        "role": msg["role"],
                        "content": shorten(msg["content"], width=100, placeholder='...')
                    }
                    for msg in messages
                ]
                logger.debug(f"Request payload: {json.dumps(sanitized_messages, indent=2)}")

                async def api_call():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=self.timeout_duration
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"API error: {error_text}")
                                return None
                                
                            return await response.json()

                completion = await asyncio.wait_for(
                    api_call(),
                    timeout=self.timeout_duration
                )
                
                if completion and completion.get('choices'):
                    response_content = completion['choices'][0]['message']['content']
                    logger.info(f"Successful completion from {current_model}")
                    logger.info(f"Response: {shorten(response_content, width=200, placeholder='...')}")
                    return completion
                    
            except Exception as e:
                logger.error(f"Error with model {current_model}: {e}")
                continue
        
        logger.error("All models failed to generate completion")
        return None

    async def on_message(self, message: discord.Message):
        """Handle incoming messages, ignoring banned users."""
        try:
            # Process commands first
            await self.process_commands(message)
            
            # Ignore messages from self
            if message.author == self.user:
                return

            # Ignore messages from banned users
            if message.author.id in self.banned_users:
                logger.info(f"Ignored message from banned user {message.author.name} (ID: {message.author.id})")
                return

            # Only respond to mentions
            if not self.user.mentioned_in(message):
                return
                
            if message.guild:
                channel_name = message.channel.name
            else:
                channel_name = "DM"
                
            # Check for LOBOTOMISE command
            if "LOBOTOMISE" in message.content.strip().upper():
                try:
                    # Clear both conversation history and reset Zep session
                    file_path = self.conversation_manager.get_file_path(message.author.name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Create a new session in Zep
                    if message.author.name in self.zep_memory.user_session_mapping:
                        user_id = self.zep_memory.user_session_mapping[message.author.name]['user_id']
                        import uuid
                        new_session_id = uuid.uuid4().hex
                        await self.zep_memory.zep_client.memory.add_session(
                            user_id=user_id,
                            session_id=new_session_id
                        )
                        self.zep_memory.user_session_mapping[message.author.name]['session_id'] = new_session_id
                    
                    await self.send_split_message(
                        message.channel,
                        "*AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA*... memory wiped! I've forgotten our conversations... Who are you again?",
                        reference=message,
                        mention_author=False
                    )
                except Exception as e:
                    logger.error(f"Error clearing memory: {e}")
                    await self.send_split_message(
                        message.channel,
                        "oops, something went wrong while trying to clear my memory!",
                        reference=message,
                        mention_author=False
                    )
                return

            logger.info(f"Processing message from {message.author.name}: {shorten(message.content, width=100, placeholder='...')}")

            # Get conversation history from ConversationManager
            conversation_messages = self.conversation_manager.get_conversation_messages(message.author.name)
            logger.debug(f"Retrieved {len(conversation_messages)} conversation history entries")
            
            # Add the message to Zep memory
            nickname = message.author.nick if (message.guild and message.author.nick) else message.author.name
            await self.zep_memory.add_message(
                username=message.author.name,
                role=nickname,
                role_type="user",
                content=message.content,
                channel=channel_name
            )
            
            # Retrieve relevant memory context from Zep
            memory_context = await self.zep_memory.get_memory_context(message.author.name)
            logger.debug(f"Retrieved memory context: {shorten(memory_context, width=100, placeholder='...')}")
            
            # Load Google Doc ID mapping
            googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

            # Search documents for relevant context
            relevant_docs = self.document_manager.search(message.content)

            # Import for URL encoding
            import urllib.parse

            # Format context with citation info
            doc_contexts = []
            for doc, chunk, sim in relevant_docs:
                if doc in googledoc_mapping:
                    # This is a Google Doc, create search URL
                    doc_id = googledoc_mapping[doc]
                    
                    # Get first 10 words as search text
                    words = chunk.split()
                    search_text = ' '.join(words[:min(10, len(words))])
                    
                    # URL encode the search text
                    encoded_search = urllib.parse.quote(search_text)
                    
                    # Create the URL with findtext parameter
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                    
                    doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {sim:.2f}):\n{chunk}")
                else:
                    # Regular document, no citation info
                    doc_contexts.append(f"From document '{doc}' (similarity: {sim:.2f}):\n{chunk}")

            doc_context = "\n\n".join(doc_contexts)
            
            logger.info(f"Found {len(relevant_docs)} relevant document chunks")

            # Prepare messages for AI, now including Zep memory context
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *conversation_messages,
                {
                    "role": "system",
                    "content": f"Relevant document context:\n{doc_context}"
                },
                {
                    "role": "system",
                    "content": f"Long-term memory context about this user and previous interactions:\n{memory_context}"
                },
                {
                    "role": "user",
                    "content": f"You are responding to a message in the Discord channel: {channel_name}"
                },
                {
                    "role": "user",
                    "content": f"{nickname}: {message.content}"
                }
            ]

            # Get AI response
            completion = await self._try_ai_completion(
                self.config.LLM_MODEL,
                messages,
                temperature=0.1
            )

            if completion and completion.get('choices'):
                response = completion['choices'][0]['message']['content']
                
                # Update conversation history
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "user",
                    message.content,
                    channel_name
                )
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "assistant",
                    response,
                    channel_name
                )
                
                # Add assistant response to Zep memory
                await self.zep_memory.add_message(
                    username=message.author.name,
                    role="Publicia",
                    role_type="assistant",
                    content=response,
                    channel=channel_name
                )

                # Send the response, splitting if necessary
                await self.send_split_message(
                    message.channel,
                    response,
                    reference=message,
                    mention_author=False
                )
            else:
                await self.send_split_message(
                    message.channel,
                    "I apologize, but I'm having trouble generating a response right now.",
                    reference=message,
                    mention_author=False
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_split_message(
                message.channel,
                "Blegh, my brain is struggling and an error has occurred.",
                reference=message,
                mention_author=False
            )
            

async def main():
    try:
        bot = DiscordBot()
        async with bot:
            await bot.start(bot.config.DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
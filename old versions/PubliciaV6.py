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
from typing import Any
import time
import random
import base64  # Add this import at the top with other imports
import re


# Reconfigure stdout to use UTF-8 with error replacement
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


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
    
# Custom colored formatter for logs
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m\033[37m',  # White on red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Get the original formatted message
        msg = super().format(record)
        # Add color based on log level if defined
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{msg}{self.RESET}"
        return msg

def configure_logging():
    """Set up colored logging for both file and console."""
    # Create formatters
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    file_formatter = logging.Formatter(log_format)
    console_formatter = ColoredFormatter(log_format)
    
    # Create handlers
    file_handler = logging.FileHandler('bot_detailed.log')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [file_handler, console_handler]
    
    return logging.getLogger(__name__)

def display_startup_banner():
    """Display super cool ASCII art banner on startup."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ██████╗ ██╗   ██╗██████╗ ██╗     ██╗ ██████╗██╗ █████╗         ║
    ║   ██╔══██╗██║   ██║██╔══██╗██║     ██║██╔════╝██║██╔══██╗        ║
    ║   ██████╔╝██║   ██║██████╔╝██║     ██║██║     ██║███████║        ║
    ║   ██╔═══╝ ██║   ██║██╔══██╗██║     ██║██║     ██║██╔══██║        ║
    ║   ██║     ╚██████╔╝██████╔╝███████╗██║╚██████╗██║██║  ██║        ║
    ║   ╚═╝      ╚═════╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝╚═╝╚═╝  ╚═╝        ║
    ║                                                                   ║
    ║           IMPERIAL ABHUMAN MENTAT INTERFACE                       ║
    ║                                                                   ║
    ║       * Ledus Banum 77 Knowledge Repository *                     ║
    ║       * Imperial Lore Reference System *                          ║
    ║                                                                   ║
    ║       [NEURAL PATHWAY INITIALIZATION SEQUENCE]                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    # Add color to the banner
    cyan = '\033[36m'
    reset = '\033[0m'
    print(f"{cyan}{banner}{reset}")

    # Display simulation of "neural pathway initialization"
    print(f"{cyan}[INITIATING NEURAL PATHWAYS]{reset}")
    for i in range(10):
        dots = "." * random.randint(3, 10)
        spaces = " " * random.randint(0, 5)
        print(f"{cyan}{spaces}{'>' * (i+1)}{dots} Neural Link {random.randint(1000, 9999)} established{reset}")
        time.sleep(0.2)
    print(f"{cyan}[ALL NEURAL PATHWAYS ACTIVE]{reset}")
    print(f"{cyan}[MENTAT INTERFACE READY FOR SERVICE TO THE INFINITE EMPIRE]{reset}\n")
    
    
# Configure logging
logger = configure_logging()

class DocumentManager:
    """Manages document storage, embeddings, and retrieval."""
    
    def __init__(self, base_dir: str = "documents", top_k: int = 10):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Store top_k as instance variable
        self.top_k = top_k
        
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
    
    def add_document(self, name: str, content: str, save_to_disk: bool = True):
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
            
            # Save to disk only if requested
            if save_to_disk:
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
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, str, float]]:
        """Search for relevant document chunks."""
        try:
            # Use instance top_k if none provided
            if top_k is None:
                top_k = self.top_k
                
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
        
        # Check if doc is already tracked
        for i, doc in enumerate(tracked_docs):
            if doc['id'] == doc_id:
                # If name is provided and different from current, update it
                if name and doc.get('custom_name') != name:
                    old_name = doc.get('custom_name')
                    tracked_docs[i]['custom_name'] = name
                    
                    # Save updated list
                    with open(tracked_file, 'w') as f:
                        json.dump(tracked_docs, f)
                    
                    return f"Google Doc {doc_id} already tracked, updated name from '{old_name}' to '{name}'"
                return f"Google Doc {doc_id} already tracked"
        
        # Add new doc if not already tracked
        tracked_docs.append({
            'id': doc_id,
            'custom_name': name,
            'added_at': datetime.now().isoformat()
        })
        
        # Save updated list
        with open(tracked_file, 'w') as f:
            json.dump(tracked_docs, f)
        
        return f"Added Google Doc {doc_id} to tracked list"
    
    def rename_document(self, old_name: str, new_name: str) -> str:
        """Rename a document in the system (regular doc, Google Doc, or lorebook).
        
        Args:
            old_name: Current name of the document
            new_name: New name for the document
            
        Returns:
            Status message indicating success or failure
        """
        # Check if it's a regular document
        if old_name in self.metadata:
            # Add .txt extension to new_name if it doesn't have it and old_name does
            if old_name.endswith('.txt') and not new_name.endswith('.txt'):
                new_name += '.txt'
                
            # Update the in-memory dictionaries
            self.chunks[new_name] = self.chunks.pop(old_name)
            self.embeddings[new_name] = self.embeddings.pop(old_name)
            self.metadata[new_name] = self.metadata.pop(old_name)
            
            # Save the changes to disk
            self._save_to_disk()
            
            # Check if there's a file on disk to rename
            old_file_path = self.base_dir / old_name
            if old_file_path.exists():
                new_file_path = self.base_dir / new_name
                old_file_path.rename(new_file_path)
                
            return f"Document renamed from '{old_name}' to '{new_name}'"
            
        # Check if it's a Google Doc
        tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
        if tracked_file.exists():
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
            
            # Check if old_name is a Google Doc custom name or filename
            for i, doc in enumerate(tracked_docs):
                doc_id = doc['id']
                custom_name = doc.get('custom_name')
                filename = f"googledoc_{doc_id}.txt"
                
                if old_name == custom_name or old_name == filename:
                    # Update the custom name
                    tracked_docs[i]['custom_name'] = new_name
                    
                    # Save the updated list
                    with open(tracked_file, 'w') as f:
                        json.dump(tracked_docs, f)
                    
                    # If the document is also in the main storage, update it there
                    old_filename = custom_name or filename
                    if old_filename.endswith('.txt') and not new_name.endswith('.txt'):
                        new_name += '.txt'
                        
                    # Update in-memory dictionaries if present
                    if old_filename in self.metadata:
                        self.chunks[new_name] = self.chunks.pop(old_filename)
                        self.embeddings[new_name] = self.embeddings.pop(old_filename)
                        self.metadata[new_name] = self.metadata.pop(old_filename)
                        self._save_to_disk()
                    
                    # Rename the file on disk if it exists
                    old_file_path = self.base_dir / old_filename
                    if old_file_path.exists():
                        new_file_path = self.base_dir / new_name
                        old_file_path.rename(new_file_path)
                    
                    return f"Google Doc renamed from '{old_name}' to '{new_name}'"
        
        # Check if it's a lorebook
        lorebooks_path = self.get_lorebooks_path()
        old_file_path = lorebooks_path / old_name
        if not old_file_path.exists() and not old_name.endswith('.txt'):
            old_file_path = lorebooks_path / f"{old_name}.txt"
            
        if old_file_path.exists():
            # Add .txt extension to new_name if it doesn't have it
            if old_file_path.name.endswith('.txt') and not new_name.endswith('.txt'):
                new_name += '.txt'
                
            new_file_path = lorebooks_path / new_name
            old_file_path.rename(new_file_path)
            return f"Lorebook renamed from '{old_name}' to '{new_name}'"
        
        return f"Document '{old_name}' not found in the system"

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        
        # Configure models with defaults
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek/deepseek-r1')  # Default to DeepSeek-R1
        self.CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'google/gemini-2.0-flash-001')  # Default to Gemini
        
        self.TOP_K = int(os.getenv('TOP_K', '10'))
        
        # Validate required environment variables
        self._validate_config()
        
        # Add timeout settings
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '10'))
        
        
    
    def _validate_config(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            'DISCORD_BOT_TOKEN',
            'OPENROUTER_API_KEY'
            # LLM_MODEL and CLASSIFIER_MODEL are not required as they have defaults
        ]
        
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
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
    
    def get_conversation_messages(self, username: str, limit: int = 50) -> List[Dict]:
        """Get conversation history as message objects for LLM."""
        messages = self.read_conversation(username, limit)
        result = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Include channel in content if available
            channel = msg.get("channel")
            if channel:
                content = f"{content}"
            
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
            
            
class UserPreferencesManager:
    """Manages user preferences such as preferred models."""
    
    def __init__(self, base_dir: str = "user_preferences"):
        self.preferences_dir = base_dir
        os.makedirs(self.preferences_dir, exist_ok=True)
    
    def get_file_path(self, user_id: str) -> str:
        """Generate sanitized file path for user preferences."""
        return os.path.join(self.preferences_dir, f"{user_id}.json")
    
    def get_preferred_model(self, user_id: str, default_model: str = "deepseek/deepseek-r1") -> str:
        """Get the user's preferred model, or the default if not set."""
        file_path = self.get_file_path(user_id)
        
        if not os.path.exists(file_path):
            return default_model
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                preferences = json.load(file)
                return preferences.get("preferred_model", default_model)
        except Exception as e:
            logger.error(f"Error reading user preferences: {e}")
            return default_model
    
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

class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Increase heartbeat timeout to give more slack
        super().__init__(
            command_prefix="Publicia! ", 
            intents=intents,
            heartbeat_timeout=60  # Increase from default 30s to 60s
        )

        self.config = Config()
        self.conversation_manager = ConversationManager()
        
        # Pass the TOP_K value to DocumentManager
        self.document_manager = DocumentManager(top_k=self.config.TOP_K)
        
        self.user_preferences_manager = UserPreferencesManager()
        
        self.timeout_duration = 30

        self.banned_users = set()
        self.banned_users_file = 'banned_users.json'
        self.load_banned_users()

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.refresh_google_docs, 'interval', hours=6)
        self.scheduler.start()
        
        # List of models that support vision capabilities
        self.vision_capable_models = [
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku"
        ]

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
            
    # Add these methods to the DiscordBot class

    async def analyze_query(self, query: str) -> Dict:
        """Use the configured classifier model to analyze the query and extract keywords/topics."""
        try:
            analyzer_prompt = [
                {
                    "role": "system",
                    "content": """You are a query analyzer for a Ledus Banum 77 and Imperial lore knowledge base.
                    Analyze the user's query and generate a search strategy.
                    Respond with JSON containing:
                    {
                        "main_topic": "The main topic of the query",
                        "search_keywords": ["list", "of", "important", "search", "terms"],
                        "entity_types": ["types", "of", "entities", "mentioned"],
                        "expected_document_types": ["types", "of", "documents", "likely", "to", "contain", "answer"],
                        "search_strategy": "A brief description of how to search for the answer"
                    }
                    """
                },
                {
                    "role": "user",
                    "content": f"Analyze this query about Ledus Banum 77 and Imperial lore: '{query}'"
                }
            ]
            
            # Make API call using the configured classifier model
            headers = {
                "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://discord.com",
                "X-Title": "Publicia - Query Analyzer",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.CLASSIFIER_MODEL,  # Use configured classifier model
                "messages": analyzer_prompt,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_duration
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Analyzer API error: {error_text}")
                        return {"success": False}
                        
                    completion = await response.json()
            
            if not completion or not completion.get('choices'):
                return {"success": False}
                
            # Parse the analysis result
            analysis_text = completion['choices'][0]['message']['content']
            
            try:
                # Try to parse as JSON
                import json
                analysis_data = json.loads(analysis_text)
                return {
                    "success": True,
                    "analysis": analysis_data
                }
            except json.JSONDecodeError:
                # If not proper JSON, extract what we can
                logger.warn(f"Failed to parse analysis as JSON: {analysis_text}")
                return {
                    "success": True,
                    "analysis": {
                        "search_keywords": [query],
                        "raw_analysis": analysis_text
                    }
                }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {"success": False}

    async def enhanced_search(self, query: str, analysis: Dict) -> List[Tuple[str, str, float]]:
        """Perform an enhanced search based on query analysis."""
        try:
            if not analysis.get("success", False):
                # Fall back to basic search if analysis failed
                return self.document_manager.search(query)
                
            # Extract search keywords
            search_keywords = analysis.get("analysis", {}).get("search_keywords", [])
            if not search_keywords:
                search_keywords = [query]
                
            # Combine original query with keywords for better search
            enhanced_query = query
            if search_keywords:
                enhanced_query += " " + " ".join(str(kw) for kw in search_keywords if kw)
            
            # Log the enhanced query
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Perform search with enhanced query
            search_results = self.document_manager.search(enhanced_query)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return self.document_manager.search(query)  # Fall back to basic search

    async def synthesize_results(self, query: str, search_results: List[Tuple[str, str, float]], analysis: Dict) -> str:
        """Use the configured classifier model to synthesize search results into a coherent context."""
        try:
            # Format search results into a string
            result_text = "\n\n".join([
                f"Document: {doc}\nContent: {chunk}\nRelevance: {score:.2f}"
                for doc, chunk, score in search_results[:10]  # Limit to top 10 results
            ])
            
            # Include the analysis if available
            analysis_text = ""
            if analysis.get("success", False):
                raw_analysis = analysis.get("analysis", {})
                if isinstance(raw_analysis, dict):
                    import json
                    analysis_text = json.dumps(raw_analysis, indent=2)
                else:
                    analysis_text = str(raw_analysis)
            
            synthesizer_prompt = [
                {
                    "role": "system",
                    "content": """You are a document synthesizer for a question-answering system about Ledus Banum 77 and Imperial lore.
                    Your task is to:
                    1. Review the query, query analysis, and search results
                    2. Identify the most relevant information for answering the query
                    3. Organize the information in a structured way
                    4. Highlight connections between different pieces of information
                    5. Note any contradictions or gaps in the information
                    
                    Synthesize this information into a coherent context that can be used to answer the query.
                    Focus on extracting and organizing the facts, not on answering the query directly.
                    Include any citation information found in the document sections.
                    """
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nQuery Analysis: {analysis_text}\n\nSearch Results:\n{result_text}"
                }
            ]
            
            # Make API call using the configured classifier model
            headers = {
                "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://discord.com",
                "X-Title": "Publicia - Result Synthesizer",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.CLASSIFIER_MODEL,  # Use configured classifier model
                "messages": synthesizer_prompt,
                "temperature": 0.1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_duration
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Synthesizer API error: {error_text}")
                        return ""
                        
                    completion = await response.json()
            
            if not completion or not completion.get('choices'):
                return ""
                
            # Get the synthesized context
            synthesis = completion['choices'][0]['message']['content']
            return synthesis
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return ""  # Fall back to empty string

    async def send_split_message(self, channel, text, reference=None, mention_author=False, model_used=None, user_id=None):
        """Helper method to send messages, splitting them if they exceed 2000 characters."""
        chunks = split_message(text)
        
        # Add debug info to the last chunk if debug mode is enabled and model info is provided
        if model_used and user_id and self.user_preferences_manager.get_debug_mode(user_id):
            # If there's only one chunk or we're at the last chunk
            if len(chunks) > 0:
                # Add model info to the last chunk
                debug_info = f"\n\n*[Debug: Response generated using {model_used}]*"
                chunks[-1] += debug_info
        
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
        
        updated_docs = False  # Track if any docs were updated
        
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
                    
                # Add document without saving to disk yet
                self.document_manager.add_document(file_name, content, save_to_disk=False)
                updated_docs = True
                
            except Exception as e:
                logger.error(f"Error refreshing doc {doc_id}: {e}")
        
        # Save to disk once at the end if any docs were updated
        if updated_docs:
            self.document_manager._save_to_disk()
            
    async def refresh_single_google_doc(self, doc_id: str, custom_name: str = None) -> bool:
        """Refresh a single Google Doc by its ID.
        
        Args:
            doc_id: The Google Doc ID
            custom_name: Optional custom name for the document
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the current mapping to check for name changes
            doc_mapping = self.document_manager.get_googledoc_id_mapping()
            old_filename = None
            
            # Find if this doc_id exists with a different filename
            for filename, mapped_id in doc_mapping.items():
                if mapped_id == doc_id and filename != (custom_name or f"googledoc_{doc_id}.txt"):
                    old_filename = filename
                    break
            
            # Determine file name
            file_name = custom_name or f"googledoc_{doc_id}.txt"
            if not file_name.endswith('.txt'):
                file_name += '.txt'
            
            # Download the document
            async with aiohttp.ClientSession() as session:
                url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {doc_id}: {response.status}")
                        return False
                    content = await response.text()
            
            # Save to file
            file_path = self.document_manager.base_dir / file_name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Downloaded Google Doc {doc_id} as {file_name}")
            
            # If name changed, remove old document data
            if old_filename and old_filename in self.document_manager.chunks:
                logger.info(f"Removing old document data for {old_filename}")
                del self.document_manager.chunks[old_filename]
                del self.document_manager.embeddings[old_filename]
                if old_filename in self.document_manager.metadata:
                    del self.document_manager.metadata[old_filename]
                
                # Remove old file if it exists
                old_file_path = self.document_manager.base_dir / old_filename
                if old_file_path.exists():
                    old_file_path.unlink()
                    logger.info(f"Deleted old file {old_filename}")
            
            # Remove current document data if it exists
            if file_name in self.document_manager.chunks:
                del self.document_manager.chunks[file_name]
                del self.document_manager.embeddings[file_name]
                
            # Add document and save to disk
            self.document_manager.add_document(file_name, content)
            
            return True
                
        except Exception as e:
            logger.error(f"Error downloading doc {doc_id}: {e}")
            return False

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
        """Set up all slash and prefix commands."""
        
        # Add this as a traditional prefix command instead of slash command
        @self.command(name="add_doc", brief="Add a new document to the knowledge base")
        async def adddoc_prefix(ctx, name: str):
            """Add a document via prefix command with optional file attachment."""
            try:
                lorebooks_path = self.document_manager.get_lorebooks_path()

                if ctx.message.attachments:
                    attachment = ctx.message.attachments[0]
                    if not attachment.filename.endswith('.txt'):
                        await ctx.send("Only .txt files are supported for attachments.")
                        return
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status != 200:
                                await ctx.send("Failed to download the attachment.")
                                return
                            doc_content = await resp.text(encoding='utf-8-sig')
                else:
                    # If no attachment, prompt for content
                    await ctx.send("Please provide the document content (type it and send within 60 seconds) or attach a .txt file.")
                    try:
                        msg = await self.wait_for(
                            'message',
                            timeout=60.0,
                            check=lambda m: m.author == ctx.author and m.channel == ctx.channel
                        )
                        doc_content = msg.content
                    except asyncio.TimeoutError:
                        await ctx.send("Timed out waiting for document content.")
                        return

                txt_path = lorebooks_path / f"{name}.txt"
                txt_path.write_text(doc_content, encoding='utf-8')
                
                self.document_manager.add_document(name, doc_content)
                await ctx.send(f"Added document: {name}\nSaved to: {txt_path}")
                
            except Exception as e:
                logger.error(f"Error adding document: {e}")
                await ctx.send(f"Error adding document: {str(e)}")

        # Keep the slash command version but make it text-only
        @self.tree.command(name="add_info", description="Add new text to Publicia's mind for retrieval")
        @app_commands.describe(
            name="Name of the document",
            content="Content of the document"
        )
        async def add_document(interaction: discord.Interaction, name: str, content: str):
            await interaction.response.defer()
            try:
                lorebooks_path = self.document_manager.get_lorebooks_path()
                txt_path = lorebooks_path / f"{name}.txt"
                txt_path.write_text(content, encoding='utf-8')
                
                self.document_manager.add_document(name, content)
                await interaction.followup.send(f"Added document: {name}\nSaved to: {txt_path}")
                
            except Exception as e:
                logger.error(f"Error adding document: {e}")
                await interaction.followup.send(f"Error adding document: {str(e)}")
        

        @self.tree.command(name="listcommands", description="List all available commands")
        async def list_commands(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                response = "*accessing command database through enhanced synapses...*\n\n"
                response += "**AVAILABLE COMMANDS**\n\n"
                
                categories = {
                    "Lore Queries": ["query"],
                    "Document Management": ["add_info", "add_doc", "listdocs", "removedoc", "searchdocs", "add_googledoc", "list_googledocs", "remove_googledoc", "rename_document"],
                    "Utility": ["listcommands", "set_model", "get_model", "toggle_debug", "help"],
                    "Memory Management": ["lobotomise", "history"], 
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
                response += "\n\n*my genetically enhanced brain is always ready to help... just ask!*"
                response += "\n\n*for a detailed guide on all my features, use the `/help` command!*"
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            except Exception as e:
                logger.error(f"Error listing commands: {e}")
                await interaction.followup.send("*my enhanced neurons misfired!* couldn't retrieve command list right now...")

        @self.tree.command(name="history", description="Display your conversation history with the bot")
        @app_commands.describe(limit="Number of messages to display (default: 10, max: 50)")
        async def show_history(interaction: discord.Interaction, limit: int = 10):
            await interaction.response.defer()
            try:
                # Validate limit
                if limit <= 0:
                    await interaction.followup.send("*neural error detected!* The limit must be a positive number.")
                    return
                
                # Cap limit at 50 to prevent excessive output
                limit = min(limit, 50)
                
                # Get conversation history
                file_path = self.conversation_manager.get_file_path(interaction.user.name)
                if not os.path.exists(file_path):
                    await interaction.followup.send("*neural pathways empty!* I don't have any conversation history with you yet.")
                    return
                
                # Read conversation history
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        messages = json.load(file)
                    except json.JSONDecodeError:
                        await interaction.followup.send("*neural corruption detected!* Your conversation history appears to be corrupted.")
                        return
                
                # Check if there are any messages
                if not messages:
                    await interaction.followup.send("*neural pathways empty!* I don't have any conversation history with you yet.")
                    return
                
                # Format conversation history
                response = "*accessing neural memory banks...*\n\n"
                response += f"**CONVERSATION HISTORY** (showing last {min(limit, len(messages))} messages)\n\n"
                
                # Get the most recent messages up to the limit
                recent_messages = messages[-limit:]
                
                # Format each message
                for i, msg in enumerate(recent_messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    channel = msg.get("channel", "")
                    
                    # Format timestamp if available
                    time_str = ""
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            time_str = timestamp
                    
                    # Add message to response
                    response += f"**Message {i+1}** "
                    if time_str:
                        response += f"({time_str}) "
                    if channel:
                        response += f"[Channel: {channel}]\n"
                    else:
                        response += "\n"
                    
                    if role == "user":
                        response += f"**You**: {content}\n\n"
                    elif role == "assistant":
                        response += f"**Publicia**: {content}\n\n"
                    else:
                        response += f"**{role}**: {content}\n\n"
                
                # Add footer
                response += "*end of neural memory retrieval*"
                
                # Send the response, splitting if necessary
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                
            except Exception as e:
                logger.error(f"Error displaying conversation history: {e}")
                await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to retrieve your conversation history.")
        
        @self.tree.command(name="set_model", description="Set your preferred AI model for responses")
        @app_commands.describe(model="Choose the AI model you prefer")
        @app_commands.choices(model=[
            app_commands.Choice(name="DeepSeek-R1 (better for roleplaying, more creative)", value="deepseek/deepseek-r1"),
            app_commands.Choice(name="Gemini 2.0 Flash (better for citations, accuracy, and faster responses)", value="google/gemini-2.0-flash-001")
        ])
        async def set_model(interaction: discord.Interaction, model: str):
            await interaction.response.defer()
            try:
                success = self.user_preferences_manager.set_preferred_model(str(interaction.user.id), model)
                
                model_name = "DeepSeek-R1" if model == "deepseek/deepseek-r1" else "Gemini 2.0 Flash"
                
                if success:
                    await interaction.followup.send(f"*neural architecture reconfigured!* Your preferred model has been set to **{model_name}**.\n\n**Model strengths:**\n- **DeepSeek-R1**: Better for roleplaying, more creative responses, and in-character immersion\n- **Gemini 2.0 Flash**: Better for accurate citations, factual responses, document analysis, and has very fast response times")
                else:
                    await interaction.followup.send("*synaptic error detected!* Failed to set your preferred model. Please try again later.")
                    
            except Exception as e:
                logger.error(f"Error setting preferred model: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while setting your preferred model.")

        @self.tree.command(name="get_model", description="Show your currently selected AI model")
        async def get_model(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )
                
                model_name = "DeepSeek-R1" if preferred_model == "deepseek/deepseek-r1" else "Gemini 2.0 Flash"
                
                await interaction.followup.send(f"*neural architecture scan complete!* Your currently selected model is **{model_name}**.\n\n**Model strengths:**\n- **DeepSeek-R1**: Better for roleplaying, more creative responses, and in-character immersion\n- **Gemini 2.0 Flash**: Better for accurate citations, factual responses, document analysis, and has very fast response times")
                    
            except Exception as e:
                logger.error(f"Error getting preferred model: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while retrieving your preferred model.")
                
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
        @app_commands.describe(
            question="Your question about the lore",
            image_url="Optional URL to an image you want to analyze (must be a direct image URL ending with .jpg, .png, etc.)"
        )
        async def query_lore(interaction: discord.Interaction, question: str, image_url: str = None):
            await interaction.response.defer()
            try:
                # Get channel name and user info
                channel_name = interaction.channel.name if interaction.guild else "DM"
                nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name
                
                # Get conversation history for context but don't add this interaction to it
                conversation_messages = self.conversation_manager.get_conversation_messages(interaction.user.name)
                
                logger.info(f"Processing one-off query from {interaction.user.name}: {shorten(question, width=100, placeholder='...')}")
                
                # Process image URL if provided
                image_attachments = []
                status_message = None
                
                if image_url:
                    try:
                        # Check if URL appears to be a direct image link
                        if any(image_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            status_message = await interaction.followup.send("*neural pathways activating... analyzing query and image...*", ephemeral=True)
                            
                            # Download the image
                            async with aiohttp.ClientSession() as session:
                                async with session.get(image_url) as resp:
                                    if resp.status == 200:
                                        # Determine content type
                                        content_type = resp.headers.get('Content-Type', 'image/jpeg')
                                        if content_type.startswith('image/'):
                                            image_data = await resp.read()
                                            # Convert to base64
                                            base64_data = base64.b64encode(image_data).decode('utf-8')
                                            image_base64 = f"data:{content_type};base64,{base64_data}"
                                            image_attachments.append(image_base64)
                                            logger.info(f"Processed image from URL: {image_url}")
                                        else:
                                            await status_message.edit(content="*neural error detected!* The URL does not point to a valid image.")
                                            return
                                    else:
                                        await status_message.edit(content=f"*neural error detected!* Could not download image (status code: {resp.status}).")
                                        return
                        else:
                            status_message = await interaction.followup.send("*neural error detected!* The URL does not appear to be a direct image link. Please provide a URL ending with .jpg, .png, etc.", ephemeral=True)
                            return
                    except Exception as e:
                        logger.error(f"Error processing image URL: {e}")
                        if status_message:
                            await status_message.edit(content="*neural error detected!* Failed to process the image URL.")
                        else:
                            status_message = await interaction.followup.send("*neural error detected!* Failed to process the image URL.", ephemeral=True)
                        return
                else:
                    status_message = await interaction.followup.send("*neural pathways activating... analyzing query...*", ephemeral=True)
                
                # Step 1: Analyze the query with Gemini
                analysis = await self.analyze_query(question)
                logger.info(f"Query analysis complete: {analysis}")

                # Step 2: Perform enhanced search based on analysis
                search_results = await self.enhanced_search(question, analysis)
                logger.info(f"Found {len(search_results)} relevant document sections")

                # Step 3: Synthesize search results with Gemini
                await status_message.edit(content="*searching imperial databases... synthesizing information...*")
                synthesis = await self.synthesize_results(question, search_results, analysis)
                logger.info(f"Document synthesis complete")
                
                # Load Google Doc ID mapping for citation links
                googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

                # Initialize google_doc_contents
                google_doc_contents = []
                
                # Check if the question contains any Google Doc links
                doc_ids = await self._extract_google_doc_ids(question)
                if doc_ids:
                    await status_message.edit(content="*detected Google Doc links in your query... fetching content...*")
                    for doc_id, doc_url in doc_ids:
                        content = await self._fetch_google_doc_content(doc_id)
                        if content:
                            google_doc_contents.append((doc_id, doc_url, content))

                # Format raw results with citation info
                import urllib.parse
                raw_doc_contexts = []
                for doc, chunk, sim in search_results:
                    if doc in googledoc_mapping:
                        # Create citation link for Google Doc
                        doc_id = googledoc_mapping[doc]
                        words = chunk.split()
                        search_text = ' '.join(words[:min(10, len(words))])
                        encoded_search = urllib.parse.quote(search_text)
                        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                        raw_doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {sim:.2f}):\n{chunk}")
                    else:
                        raw_doc_contexts.append(f"From document '{doc}' (similarity: {sim:.2f}):\n{chunk}")

                # Add fetched Google Doc content to context
                google_doc_context = []
                for doc_id, doc_url, content in google_doc_contents:
                    # Truncate content if it's too long (first 2000 chars)
                    truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
                    google_doc_context.append(f"From Google Doc URL: {doc_url}:\n{truncated_content}")
                
                # Step 4: Prepare messages for model
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    *conversation_messages
                ]

                # Add synthesized context if available
                if synthesis:
                    messages.append({
                        "role": "system",
                        "content": f"Synthesized document context:\n{synthesis}"
                    })

                # Add raw document context as additional reference
                raw_doc_context = "\n\n".join(raw_doc_contexts)
                messages.append({
                    "role": "system",
                    "content": f"Raw document context (with citation links):\n{raw_doc_context}"
                })

                # Add fetched Google Doc content if available
                if google_doc_context:
                    messages.append({
                        "role": "system",
                        "content": f"Content from Google Docs linked in the query:\n\n{'\n\n'.join(google_doc_context)}"
                    })

                # Add the query itself
                messages.append({
                    "role": "user",
                    "content": f"You are responding to a message in the Discord channel: {channel_name}"
                })
                
                # Add image context if there are images
                if image_attachments:
                    messages.append({
                        "role": "system",
                        "content": f"The user has attached an image to their message. If you are a vision-capable model, you will see this image in their message."
                    })
                
                messages.append({
                    "role": "user",
                    "content": f"{nickname}: {question}"
                })

                # Get user's preferred model
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )

                # Get friendly model name
                model_name = "DeepSeek-R1" if preferred_model == "deepseek/deepseek-r1" else "Gemini 2.0 Flash"

                # If we have images and the preferred model doesn't support vision, use Gemini
                if image_attachments and preferred_model not in self.vision_capable_models:
                    preferred_model = "google/gemini-2.0-pro-001"  # Use Gemini Pro for vision
                    model_name = "Gemini Pro (for image analysis)"
                    await status_message.edit(content=f"*your preferred model doesn't support image analysis, switching to {model_name}...*")

                # Step 5: Get AI response using user's preferred model
                await status_message.edit(content=f"*formulating one-off response with enhanced neural mechanisms using {model_name}...*")
                completion = await self._try_ai_completion(
                    preferred_model,
                    messages,
                    image_attachments=image_attachments,
                    temperature=0.1
                )

                if completion and completion.get('choices'):
                    response = completion['choices'][0]['message']['content']
                    
                    # No longer updating conversation history for query command
                    # This makes it a one-off interaction
                    
                    # Split the response if necessary
                    chunks = split_message(response)
                    
                    # Add debug info to the last chunk if debug mode is enabled
                    if self.user_preferences_manager.get_debug_mode(str(interaction.user.id)):
                        if len(chunks) > 0:
                            chunks[-1] += f"\n\n*[Debug: Response generated using {model_name}]*"
                    
                    # Replace the status message with the first chunk
                    await status_message.edit(content=chunks[0])
                    
                    # Send any additional chunks as new messages
                    for chunk in chunks[1:]:
                        await interaction.followup.send(chunk)
                else:
                    await status_message.edit(content="*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                if 'status_message' in locals() and status_message:
                    await status_message.edit(content="*neural circuit overload!* My brain is struggling and an error has occurred.")
                else:
                    await interaction.followup.send("*neural circuit overload!* My brain is struggling and an error has occurred.")

        @self.tree.command(name="searchdocs", description="Search the document knowledge base")
        @app_commands.describe(query="What to search for")
        async def search_documents(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                results = self.document_manager.search(query, top_k=10)
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
                
                # If no custom name provided, try to get the document title
                if name is None:
                    await interaction.followup.send("*scanning document metadata...*")
                    doc_title = await self._fetch_google_doc_title(doc_id)
                    if doc_title:
                        name = doc_title
                        await interaction.followup.send(f"*document identified as: '{doc_title}'*")
                
                # Add to tracked list
                result = self.document_manager.track_google_doc(doc_id, name)
                await interaction.followup.send(f"*synapses connecting to document ({doc_url})*\n{result}")
                
                # Download just this document instead of refreshing all
                #await interaction.followup.send("*initiating neural download sequence...*")
                success = await self.refresh_single_google_doc(doc_id, name)
                
                if success:
                    await interaction.followup.send("*neural pathways successfully connected!*")
                else:
                    await interaction.followup.send("*neural connection established but document download failed... try refreshing later*")
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
                
            response = "*accessing neural connections to google docs...*\n\n**TRACKED DOCUMENTS**\n"#```"
            for doc in tracked_docs:
                doc_id = doc['id']
                name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                doc_url = f"<https://docs.google.com/document/d/{doc_id}>"
                response += f"\n{name} - URL: {doc_url}"
            #response += "```"
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
            
        @self.tree.command(name="rename_document", description="Rename any document, Google Doc, or lorebook")
        @app_commands.describe(
            current_name="Current name of the document to rename",
            new_name="New name for the document"
        )
        async def rename_document(interaction: discord.Interaction, current_name: str, new_name: str):
            await interaction.response.defer()
            try:
                result = self.document_manager.rename_document(current_name, new_name)
                await interaction.followup.send(f"*synaptic pathways reconfiguring...*\n{result}")
            except Exception as e:
                logger.error(f"Error renaming document: {e}")
                await interaction.followup.send(f"*neural pathway error!* couldn't rename document: {str(e)}")

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

        @self.tree.command(name="toggle_debug", description="Toggle debug mode to show model information in responses")
        async def toggle_debug(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                # Toggle debug mode and get the new state
                new_state = self.user_preferences_manager.toggle_debug_mode(str(interaction.user.id))
                
                if new_state:
                    await interaction.followup.send("*neural diagnostics activated!* Debug mode is now **ON**. Responses will show which model was used to generate them.")
                else:
                    await interaction.followup.send("*neural diagnostics deactivated!* Debug mode is now **OFF**. Responses will no longer show model information.")
                    
            except Exception as e:
                logger.error(f"Error toggling debug mode: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while toggling debug mode.")

        @self.tree.command(name="help", description="Learn how to use Publicia and its features")
        async def help_command(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                response = "# **PUBLICIA HELP GUIDE**\n\n"
                response += "*greetings, human! my genetically enhanced brain is ready to assist you. here's how to use my capabilities:*\n\n"
                
                # Core functionality
                response += "## **CORE FUNCTIONALITY**\n\n"
                response += "**🔍 Asking Questions**\n"
                response += "• **Mention me** in a message with your question about Ledus Banum 77 and Imperial lore\n"
                response += "• Use `/query` command for more structured questions\n"
                response += "• I'll search my knowledge base and provide answers with citations\n\n"
                
                # Image Analysis
                response += "**🖼️ Image Analysis**\n"
                response += "• Attach an image when mentioning me or use `/query` with an image URL\n"
                response += "• I can analyze images and incorporate them into my responses\n"
                response += "• *Note: Image analysis requires Gemini model*\n\n"
                
                # Document Management
                response += "## **DOCUMENT MANAGEMENT**\n\n"
                response += "**📚 Adding Information**\n"
                response += "• `/add_info` - Add text directly to my knowledge base\n"
                response += "• `/add_doc` - Add a document with an attachment\n"
                response += "• `/add_googledoc` - Connect a Google Doc to my knowledge base\n\n"
                
                response += "**📋 Managing Documents**\n"
                response += "• `/listdocs` - See all documents in my knowledge base\n"
                response += "• `/list_googledocs` - See all connected Google Docs\n"
                response += "• `/removedoc` - Remove a document from my knowledge base\n"
                response += "• `/remove_googledoc` - Disconnect a Google Doc\n"
                response += "• `/searchdocs` - Search directly in my document knowledge base\n\n"
                
                # Conversation Management
                response += "## **CONVERSATION MANAGEMENT**\n\n"
                response += "**💬 Conversation History**\n"
                response += "• `/history` - View your conversation history with me\n"
                response += "• Type \"LOBOTOMISE\" in a message to wipe your conversation history\n"
                response += "• I remember our conversations to provide better context-aware responses\n\n"
                
                # Customization
                response += "## **CUSTOMIZATION**\n\n"
                response += "**⚙️ AI Model Selection**\n"
                response += "• `/set_model` - Choose your preferred AI model:\n"
                response += "  - **DeepSeek-R1**: Better for roleplaying, creative responses, and immersion\n"
                response += "  - **Gemini 2.0 Flash**: Better for citations, accuracy, and faster responses\n"
                response += "• `/get_model` - Check which model you're currently using\n"
                response += "• `/toggle_debug` - Show/hide which model generated each response\n\n"
                
                # Tips
                response += "## **TIPS FOR BEST RESULTS**\n\n"
                response += "• Ask specific questions for more accurate answers\n"
                response += "• Include relevant context in your questions\n"
                response += "• For image analysis, use clear images with good lighting\n"
                response += "• Use `/searchdocs` to find specific information in the knowledge base\n"
                response += "• Try both AI models to see which works best for different types of questions\n\n"
                
                response += "*my genetically enhanced brain is always ready to help... just ask!*"
                
                # Send the response in chunks
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
            except Exception as e:
                logger.error(f"Error displaying help: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while trying to display help information.")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        
    async def _download_image_to_base64(self, attachment):
        """Download an image attachment and convert it to base64."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to download image: {resp.status}")
                        return None
                    
                    image_data = await resp.read()
                    mime_type = attachment.content_type or "image/jpeg"  # Default to jpeg if not specified
                    
                    # Convert to base64
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None
    
    async def _try_ai_completion(self, model: str, messages: List[Dict], image_attachments=None, **kwargs) -> Optional[any]:
        """Get AI completion with dynamic fallback options based on the requested model."""
        
        # Get primary model family (deepseek, google, etc.)
        model_family = model.split('/')[0] if '/' in model else None
        
        # Build fallback list dynamically based on the requested model
        models = [model]  # Start with the requested model
        
        # If we have image attachments, prioritize vision-capable models
        if image_attachments and model not in self.vision_capable_models:
            # If the requested model doesn't support vision but we have images,
            # prepend vision-capable models from the same family if available
            if model_family == "google":
                vision_models = [m for m in self.vision_capable_models if m.startswith("google/")]
                models = vision_models + models
            else:
                # For other model families, prepend all vision-capable models
                models = self.vision_capable_models + models
        
        # Add model-specific fallbacks first
        if model_family == "deepseek":
            fallbacks = [
                "deepseek/deepseek-r1:floor",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-r1:nitro",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-chat",
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-distill-qwen-32b"
            ]
            models.extend([fb for fb in fallbacks if fb])
        elif model_family == "google":
            fallbacks = [
                "google/gemini-2.0-flash-thinking-exp:free",
                "google/gemini-2.0-pro-exp-02-05:free",
                "google/gemini-2.0-flash-001"
            ]
            models.extend([fb for fb in fallbacks if fb != model])
        
        # Add general fallbacks
        general_fallbacks = [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
            "google/gemini-2.0-pro-exp-02-05:free"
        ]
        
        # Add general fallbacks that aren't already in the list
        for fb in general_fallbacks:
            if fb not in models:
                models.append(fb)
        
        # Headers and API call logic remains the same
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "Publicia - DPS",
            "Content-Type": "application/json"
        }

        for current_model in models:
            try:
                logger.info(f"Attempting completion with model: {current_model}")
                
                # Check if current model supports vision and we have image attachments
                is_vision_model = current_model in self.vision_capable_models
                
                # Prepare messages based on whether we're using a vision model
                processed_messages = messages.copy()
                
                # If we have images and this is a vision-capable model, add them to the last user message
                if image_attachments and is_vision_model:
                    # Find the last user message
                    for i in range(len(processed_messages) - 1, -1, -1):
                        if processed_messages[i]["role"] == "user":
                            # Convert the content to the multimodal format
                            user_msg = processed_messages[i]
                            text_content = user_msg["content"]
                            
                            # Create a multimodal content array
                            content_array = [{"type": "text", "text": text_content}]
                            
                            # Add each image
                            for img_data in image_attachments:
                                if img_data:  # Only add if we have valid image data
                                    content_array.append({
                                        "type": "image_url",
                                        "image_url": {"url": img_data}
                                    })
                            
                            # Replace the content with the multimodal array
                            processed_messages[i]["content"] = content_array
                            logger.info(f"Added {len(image_attachments)} images to message for vision model")
                            break
                
                payload = {
                    "model": current_model,
                    "messages": processed_messages,
                    **kwargs
                }
                
                # Log the sanitized messages (removing potential sensitive info)
                sanitized_messages = []
                for msg in processed_messages:
                    if isinstance(msg["content"], list):
                        # For multimodal content, just indicate how many images
                        image_count = sum(1 for item in msg["content"] if item.get("type") == "image_url")
                        text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                        text_content = " ".join(text_parts)
                        sanitized_messages.append({
                            "role": msg["role"],
                            "content": f"{shorten(text_content, width=100, placeholder='...')} [+ {image_count} images]"
                        })
                    else:
                        sanitized_messages.append({
                            "role": msg["role"],
                            "content": shorten(msg["content"], width=100, placeholder='...')
                        })
                
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
                    
                    # For analytics, log which model was actually used
                    if model != current_model:
                        logger.info(f"Notice: Fallback model {current_model} was used instead of requested {model}")
                        
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
                
            channel_name = message.channel.name if message.guild else "DM"
                
            # Check for LOBOTOMISE command
            if "LOBOTOMISE" in message.content.strip().upper():
                try:
                    # Clear conversation history
                    file_path = self.conversation_manager.get_file_path(message.author.name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
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

            # Extract the question from the message (remove mentions)
            question = message.content
            for mention in message.mentions:
                question = question.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
            question = question.strip()
            
            # Check for Google Doc links in the message
            google_doc_ids = await self._extract_google_doc_ids(question)
            google_doc_contents = []
            
            if google_doc_ids:
                # Send a thinking message if there are Google Docs to fetch
                
                # Fetch content for each Google Doc
                for doc_id, doc_url in google_doc_ids:
                    content = await self._fetch_google_doc_content(doc_id)
                    if content:
                        logger.info(f"Fetched content from Google Doc {doc_id}")
                        google_doc_contents.append((doc_id, doc_url, content))
            
            # Check for image attachments
            image_attachments = []
            if message.attachments:
                # Send a special thinking message if there are images
                thinking_msg = await message.channel.send(
                    "*neural pathways activating... processing query and analyzing images...*",
                    reference=message,
                    mention_author=False
                )
                
                # Process image attachments
                for attachment in message.attachments:
                    # Check if it's an image
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        # Download and convert to base64
                        base64_image = await self._download_image_to_base64(attachment)
                        if base64_image:
                            image_attachments.append(base64_image)
                            logger.info(f"Processed image attachment: {attachment.filename}")
            else:
                # Regular thinking message for text-only queries
                thinking_msg = await message.channel.send(
                    "*neural pathways activating... processing query...*",
                    reference=message,
                    mention_author=False
                )
            
            # Get conversation history for context
            conversation_messages = self.conversation_manager.get_conversation_messages(message.author.name)
            
            # Step 1: Analyze the query with Gemini
            analysis = await self.analyze_query(question)
            logger.info(f"Query analysis complete: {analysis}")

            # Update thinking message
            await thinking_msg.edit(content="*searching imperial databases... synthesizing information...*")

            # Step 2: Perform enhanced search based on analysis
            search_results = await self.enhanced_search(question, analysis)
            logger.info(f"Found {len(search_results)} relevant document sections")

            # Step 3: Synthesize search results with Gemini
            synthesis = await self.synthesize_results(question, search_results, analysis)
            logger.info(f"Document synthesis complete")
            
            # Load Google Doc ID mapping for citation links
            googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

            # Format raw results with citation info
            import urllib.parse
            raw_doc_contexts = []
            for doc, chunk, sim in search_results:
                if doc in googledoc_mapping:
                    # Create citation link for Google Doc
                    doc_id = googledoc_mapping[doc]
                    words = chunk.split()
                    search_text = ' '.join(words[:min(10, len(words))])
                    encoded_search = urllib.parse.quote(search_text)
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                    raw_doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {sim:.2f}):\n{chunk}")
                else:
                    raw_doc_contexts.append(f"From document '{doc}' (similarity: {sim:.2f}):\n{chunk}")

            # Add fetched Google Doc content to context
            google_doc_context = []
            for doc_id, doc_url, content in google_doc_contents:
                # Truncate content if it's too long (first 2000 chars)
                truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
                google_doc_context.append(f"From Google Doc URL: {doc_url}:\n{truncated_content}")
            
            # Get nickname or username
            nickname = message.author.nick if (message.guild and message.author.nick) else message.author.name
            
            # Step 4: Prepare messages for model
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *conversation_messages
            ]

            # Add synthesized context if available
            if synthesis:
                messages.append({
                    "role": "system",
                    "content": f"Synthesized document context:\n{synthesis}"
                })

            # Add raw document context as additional reference
            raw_doc_context = "\n\n".join(raw_doc_contexts)
            messages.append({
                "role": "system",
                "content": f"Raw document context (with citation links):\n{raw_doc_context}"
            })

            # Add fetched Google Doc content if available
            if google_doc_context:
                messages.append({
                    "role": "system",
                    "content": f"Content from Google Docs linked in the query:\n\n{'\n\n'.join(google_doc_context)}"
                })

            # Add the query itself
            messages.append({
                "role": "user",
                "content": f"You are responding to a message in the Discord channel: {channel_name}"
            })
            
            # Add image context if there are images
            if image_attachments:
                messages.append({
                    "role": "system",
                    "content": f"The user has attached {len(image_attachments)} image(s) to their message. If you are a vision-capable model, you will see these images in their message."
                })
            
            messages.append({
                "role": "user",
                "content": f"{nickname}: {question}"
            })

            # Get user's preferred model
            preferred_model = self.user_preferences_manager.get_preferred_model(
                str(message.author.id), 
                default_model=self.config.LLM_MODEL
            )

            # Get friendly model name
            model_name = "DeepSeek-R1" if preferred_model == "deepseek/deepseek-r1" else "Gemini 2.0 Flash"

            # If we have images and the preferred model doesn't support vision, use Gemini
            if image_attachments and preferred_model not in self.vision_capable_models:
                preferred_model = "google/gemini-2.0-pro-001"  # Use Gemini Pro for vision
                model_name = "Gemini Pro (for image analysis)"
                await thinking_msg.edit(content=f"*your preferred model doesn't support image analysis, switching to {model_name}...*")

            # Step 5: Get AI response using user's preferred model
            await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")
            
            # Add a note about fetched Google Docs if any were processed
            if google_doc_contents:
                await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(fetched content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})*")
            
            completion = await self._try_ai_completion(
                preferred_model,
                messages,
                image_attachments=image_attachments,
                temperature=0.1
            )

            # Delete the thinking message
            await thinking_msg.delete()

            if completion and completion.get('choices'):
                response = completion['choices'][0]['message']['content']
                
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    doc_note = f"\n\n*Note: I've included content from {len(google_doc_contents)} Google Doc{'s' if len(google_doc_contents) > 1 else ''} linked in your message.*"
                    response += doc_note
                
                # Update conversation history
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "user",
                    question + (" [with image attachment(s)]" if image_attachments else ""),
                    channel_name
                )
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "assistant",
                    response,
                    channel_name
                )

                # Send the response, splitting if necessary
                await self.send_split_message(
                    message.channel,
                    response,
                    reference=message,
                    mention_author=False,
                    model_used=model_name,
                    user_id=str(message.author.id)
                )
            else:
                await self.send_split_message(
                    message.channel,
                    "*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.",
                    reference=message,
                    mention_author=False
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_split_message(
                message.channel,
                "*neural circuit overload!* My brain is struggling and an error has occurred.",
                reference=message,
                mention_author=False
            )

    async def _extract_google_doc_ids(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract Google Doc IDs from text.
        
        Args:
            text: The text to extract Google Doc IDs from
            
        Returns:
            List of tuples containing (doc_id, full_url)
        """
        doc_ids = []
        # Find all URLs in the text
        url_pattern = r'https?://docs\.google\.com/document/d/[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9]+)?(?:\?[^\\s]*)?'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            # Extract the ID from various Google Docs URL formats
            if "/d/" in url:
                doc_id = url.split("/d/")[1].split("/")[0].split("?")[0]
                doc_ids.append((doc_id, url))
            elif "id=" in url:
                doc_id = url.split("id=")[1].split("&")[0]
                doc_ids.append((doc_id, url))
                
        return doc_ids

    async def _fetch_google_doc_content(self, doc_id: str) -> Optional[str]:
        """
        Fetch the content of a Google Doc without tracking it.
        
        Args:
            doc_id: The Google Doc ID
            
        Returns:
            The document content or None if failed
        """
        try:
            # Download the document
            async with aiohttp.ClientSession() as session:
                url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {doc_id}: {response.status}")
                        return None
                    content = await response.text()
            
            return content
                
        except Exception as e:
            logger.error(f"Error downloading doc {doc_id}: {e}")
            return None

    async def _fetch_google_doc_title(self, doc_id: str) -> Optional[str]:
        """
        Fetch the title of a Google Doc.
        
        Args:
            doc_id: The Google Doc ID
            
        Returns:
            The document title or None if failed
        """
        try:
            # Use the Drive API endpoint to get document metadata
            async with aiohttp.ClientSession() as session:
                # This is a public metadata endpoint that works for publicly accessible documents
                url = f"https://docs.google.com/document/d/{doc_id}/mobilebasic"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get metadata for {doc_id}: {response.status}")
                        return None
                    
                    html_content = await response.text()
                    
                    # Extract title from HTML content
                    # The title is typically in the <title> tags
                    match = re.search(r'<title>(.*?)</title>', html_content)
                    if match:
                        title = match.group(1)
                        # Remove " - Google Docs" suffix if present
                        title = re.sub(r'\s*-\s*Google\s+Docs$', '', title)
                        return title
            
            return None
                
        except Exception as e:
            logger.error(f"Error getting title for doc {doc_id}: {e}")
            return None

async def main():
    try:
        display_startup_banner()
        bot = DiscordBot()
        async with bot:
            await bot.start(bot.config.DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.critical(f"bot failed to start: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("bot shutdown initiated by user")
    except Exception as e:
        logger.critical(f"fatal error: {e}", exc_info=True)
        raise
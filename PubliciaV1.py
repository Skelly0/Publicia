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
    
    # Update the DocumentManager.search method
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
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
                logger.debug(f"Chunk content: {shorten(chunk, width=200, placeholder='...')}")
            
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
                        with open(txt_file, 'r', encoding='utf-8') as f:
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

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.LLM_MODEL = os.getenv('LLM_MODEL')
        
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
            'LLM_MODEL'
        ]
        
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class ConversationManager:
    """Manages conversation history for context."""
    
    def __init__(self, base_dir: str = "conversations"):
        self.conversation_dir = base_dir
        os.makedirs(self.conversation_dir, exist_ok=True)

    def get_file_path(self, username: str) -> str:
        """Generate sanitized file path for user conversations."""
        sanitized_username = "".join(c for c in username if c.isalnum() or c in (' ', '.', '_')).rstrip()
        return os.path.join(self.conversation_dir, f"{sanitized_username}.txt")

    def read_conversation(self, username: str, limit: int = 10) -> List[str]:
        """Read recent conversation history for a user."""
        file_path = self.get_file_path(username)
        
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return list(deque(file, maxlen=limit))
        except Exception as e:
            logger.error(f"Error reading conversation: {e}")
            return []

    def write_conversation(self, username: str, message: str):
        """Append a message to the user's conversation history."""
        try:
            with open(self.get_file_path(username), 'a', encoding='utf-8') as file:
                file.write(f"{message}\n")
        except Exception as e:
            logger.error(f"Error writing conversation: {e}")

class DiscordBot(commands.Bot):
    """Discord bot implementation with Q&A capabilities."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="Publicia! ", intents=intents)

        # Initialize configuration and components
        self.config = Config()
        self.conversation_manager = ConversationManager()
        self.document_manager = DocumentManager()
        self.timeout_duration = 30  # seconds
        
        # Register commands
        self.add_commands()
    
    def add_commands(self):
        """Add bot commands."""
        
        @self.command(name='adddoc')
        async def add_document(ctx, name: str, *, content: str):
            """Add a new document to the knowledge base."""
            try:
                self.document_manager.add_document(name, content)
                await ctx.send(f"Added document: {name}")
            except Exception as e:
                await ctx.send(f"Error adding document: {str(e)}")

    async def setup_hook(self):
        """Initial setup hook called by discord.py."""
        logger.info("Bot is setting up...")
        
    async def _try_ai_completion(self, model: str, messages: List[Dict], **kwargs) -> Optional[any]:
        """Get AI completion with fallback options."""
        
        models = [
            "deepseek/deepseek-r1:free",
            model,  # Try the requested model first
            "deepseek/deepseek-r1:floor",
            "deepseek/deepseek-r1:nitro",
            "deepseek/deepseek-r1",
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

    async def on_ready(self):
        """Handler for bot ready event."""
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        try:
            # Process commands first
            await self.process_commands(message)
            
            # Ignore messages from self
            if message.author == self.user:
                return

            # Only respond to mentions
            if not self.user.mentioned_in(message):
                return
                
            # Only respond to mentions
            if not self.user.mentioned_in(message):
                return
                
            # Check for LOBOTOMISE command
            if message.content.strip().upper() == "LOBOTOMISE":
                try:
                    file_path = self.conversation_manager.get_file_path(message.author.name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        await message.channel.send(
                            "*AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA*... memory wiped! I've forgotten our conversations... Who are you again? :3333",
                            reference=message,
                            mention_author=False
                        )
                    else:
                        await message.channel.send(
                            "hmm, i don't seem to have any memories of our conversations to wipe!",
                            reference=message,
                            mention_author=False
                        )
                    return
                except Exception as e:
                    logger.error(f"Error clearing conversation history: {e}")
                    await message.channel.send(
                        "oops, something went wrong while trying to clear my memory!",
                        reference=message,
                        mention_author=False
                    )
                    return

            logger.info(f"Processing message from {message.author.name}: {shorten(message.content, width=100, placeholder='...')}")

            # Get conversation history for context
            conversation = self.conversation_manager.read_conversation(message.author.name)
            context = "\n".join(conversation)
            logger.debug(f"Retrieved {len(conversation)} conversation history entries")
            
            # Search documents for relevant context
            relevant_docs = self.document_manager.search(message.content)
            doc_context = "\n\n".join([
                f"From document '{doc}' (similarity: {sim:.2f}):\n{chunk}"
                for doc, chunk, sim in relevant_docs
            ])
            
            logger.info(f"Found {len(relevant_docs)} relevant document chunks")

            # Prepare messages for AI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Q&A abhuman mentat (think Dune) named Publicia. You are NOT a robot/AI, you are biological. You specialise in Ledus Banum 77 and Imperial lore and information. "
                        "Use the provided document context to answer questions accurately. "
                        "If you're unsure or the information isn't in the documents, say so."
                        "Do not make anything up about the setting of Ledus Banum 77 or the Empire. If you don't have information on what the user is asking, say you don't know."
                    )
                },
                {
                    "role": "system",
                    "content": f"Relevant document context:\n{doc_context}"
                },
                {
                    "role": "system",
                    "content": f"Previous conversation context:\n{context}"
                },
                {
                    "role": "user",
                    "content": message.content
                }
            ]

            # Get AI response
            completion = await self._try_ai_completion(
                self.config.LLM_MODEL,
                messages,
                temperature=0.7
            )

            if completion and completion.get('choices'):
                response = completion['choices'][0]['message']['content']
                
                # Save to conversation history
                self.conversation_manager.write_conversation(
                    message.author.name,
                    f"User: {message.content}\nBot: {response}"
                )

                # Send response
                await message.channel.send(
                    response,
                    reference=message,
                    mention_author=False
                )
            else:
                await message.channel.send(
                    "I apologize, but I'm having trouble generating a response right now.",
                    reference=message,
                    mention_author=False
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.channel.send(
                "An error occurred while processing your message.",
                reference=message,
                mention_author=False
            )

    async def add_utility_commands(self):
        """Add utility commands to the bot."""
        
        @self.command(name='listdocs')
        async def list_documents(ctx):
            """List all available documents."""
            try:
                if not self.document_manager.metadata:
                    await ctx.send("No documents found in the knowledge base.")
                    return
                    
                response = "Available documents:\n```"
                for doc_name, meta in self.document_manager.metadata.items():
                    chunks = meta['chunk_count']
                    added = meta['added']
                    response += f"\n{doc_name} - {chunks} chunks (Added: {added})"
                response += "```"
                
                await ctx.send(response)
            except Exception as e:
                await ctx.send(f"Error listing documents: {str(e)}")

        @self.command(name='removedoc')
        async def remove_document(ctx, name: str):
            """Remove a document from the knowledge base."""
            try:
                if name in self.document_manager.metadata:
                    # Remove from all storage
                    del self.document_manager.chunks[name]
                    del self.document_manager.embeddings[name]
                    del self.document_manager.metadata[name]
                    
                    # Save changes
                    self.document_manager._save_to_disk()
                    
                    await ctx.send(f"Removed document: {name}")
                else:
                    await ctx.send(f"Document not found: {name}")
            except Exception as e:
                await ctx.send(f"Error removing document: {str(e)}")

        @self.command(name='searchdocs')
        async def search_documents(ctx, *, query: str):
            """Directly search the document knowledge base."""
            try:
                results = self.document_manager.search(query, top_k=3)
                
                if not results:
                    await ctx.send("No relevant documents found.")
                    return
                    
                response = "Search results:\n```"
                for doc_name, chunk, similarity in results:
                    response += f"\nFrom {doc_name} (similarity: {similarity:.2f}):\n"
                    response += f"{chunk[:200]}...\n"
                response += "```"
                
                await ctx.send(response)
            except Exception as e:
                await ctx.send(f"Error searching documents: {str(e)}")

        @self.command(name='reloaddocs')
        async def reload_documents(ctx):
            """Reload all documents and regenerate embeddings."""
            try:
                self.document_manager._load_documents()
                await ctx.send("Documents reloaded successfully.")
            except Exception as e:
                await ctx.send(f"Error reloading documents: {str(e)}")

async def main():
    """Main entry point for the Discord bot."""
    try:
        bot = DiscordBot()
        async with bot:
            # Add utility commands
            await bot.add_utility_commands()
            # Start the bot
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
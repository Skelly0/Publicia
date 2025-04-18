"""
Discord bot implementation for Publicia

This module contains the main DiscordBot class that integrates all the managers
and handles Discord events and commands.
"""
import re
import io
import os
import json
import base64
import asyncio
import logging
import random
import aiohttp
import discord
import numpy as np
from datetime import datetime
from textwrap import shorten
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set, Union # Added Union
from discord import app_commands
from discord.ext import commands
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sys

from prompts.system_prompt import SYSTEM_PROMPT, INFORMATIONAL_SYSTEM_PROMPT # Added INFORMATIONAL_SYSTEM_PROMPT
from prompts.image_prompt import IMAGE_DESCRIPTION_PROMPT
from utils.helpers import check_permissions, is_image, split_message
from utils.logging import sanitize_for_logging

from commands import (
    document_commands,
    image_commands,
    conversation_commands,
    admin_commands,
    utility_commands,
    query_commands
)

logger = logging.getLogger(__name__)

class DiscordBot(commands.Bot):
    """Discord bot for the Ledus Banum 77 and Imperial Lore Q&A system."""
    
    def __init__(self, config=None, document_manager=None, image_manager=None, 
                 conversation_manager=None, user_preferences_manager=None):
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Increase heartbeat timeout to give more slack
        super().__init__(
            command_prefix="Publicia! ", 
            intents=intents,
            heartbeat_timeout=60  # Increase from default 30s to 60s
        )

        self.config = config
        self.document_manager = document_manager
        self.image_manager = image_manager
        self.conversation_manager = conversation_manager
        self.user_preferences_manager = user_preferences_manager
        
        # Add search caching
        self.search_cache = {}  # Store previous search results by user
        
        self.timeout_duration = 500

        self.banned_users = set()
        self.banned_users_file = 'banned_users.json'
        self.load_banned_users()

        self.scheduler = AsyncIOScheduler()
        # Schedule the async function directly
        self.scheduler.add_job(self.refresh_google_docs, 'interval', hours=6) 
        self.scheduler.start()
        
        # List of models that support vision capabilities
        self.vision_capable_models = [
            "google/gemini-2.5-flash-preview",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.5-pro-exp-03-25:free", # Added new vision model
            "microsoft/phi-4-multimodal-instruct",
            "anthropic/claude-3.7-sonnet:beta",
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.5-sonnet:beta", 
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku:beta",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-haiku:beta",
            "meta-llama/llama-4-maverick:floor",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
        ]

    def sanitize_discord_text(self, text: str) -> str:
        """Sanitize text for Discord message display by escaping special characters."""
        # Replace backslashes to avoid escape sequence issues
        text = text.replace("\\", "\\\\")
        return text
    
    # Removed refresh_google_docs_wrapper as it's not needed with apscheduler running coroutines directly

    async def fetch_channel_messages(self, channel, limit: int = 20, max_message_length: int = 500) -> List[Dict]:
        """Fetch recent messages from a channel.
        
        Args:
            channel: The Discord channel to fetch messages from
            limit: Maximum number of messages to retrieve
            max_message_length: Maximum length of each message
            
        Returns:
            List of message dictionaries (author, content, timestamp)
        """
        try:
            messages = []
            async for message in channel.history(limit=limit):
                # Skip messages from the bot itself
                if message.author == self.user:
                    continue
                    
                # Truncate long messages
                content = message.content
                if len(content) > max_message_length:
                    content = content[:max_message_length] + "..."
                    
                # Format the message
                messages.append({
                    "author": message.author.nick if (message.guild and hasattr(message.author, 'nick') and message.author.nick) else message.author.name,
                    "content": content,
                    "timestamp": message.created_at.isoformat()
                })
                
            # Reverse the list to get chronological order (oldest first)
            messages.reverse()
            
            return messages
        except Exception as e:
            logger.error(f"Error fetching channel messages: {e}")
            return []

    def load_banned_users(self):
        """Load banned users from JSON file."""
        try:
            with open(self.banned_users_file, 'r') as f:
                data = json.load(f)
                self.banned_users = set(data.get('banned_users', []))
        except FileNotFoundError:
            self.banned_users = set()
        except json.JSONDecodeError:
            logger.error(f"Error decoding {self.banned_users_file}. Using empty banned users list.")
            self.banned_users = set()

    def save_banned_users(self):
        """Save banned users to JSON file."""
        try:
            with open(self.banned_users_file, 'w') as f:
                json.dump({'banned_users': list(self.banned_users)}, f)
        except Exception as e:
            logger.error(f"Error saving banned users: {e}")
            
    async def _generate_image_description(self, image_data: bytes) -> str:
        """Generate a description for an image using a vision-capable model."""
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare API call
            headers = {
                "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://discord.com",
                "X-Title": "Publicia - Image Describer",
                "Content-Type": "application/json"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": IMAGE_DESCRIPTION_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed description of this image, focusing on all visual elements and potential connections to Ledus Banum 77 or Imperial lore."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            
            payload = {
                "model": "google/gemini-2.5-flash-preview",  # Use a vision-capable model
                "messages": messages,
                "temperature": 0.1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Image description API error: {error_text}")
                        return "Failed to generate description."
                        
                    completion = await response.json()
            
            if not completion or not completion.get('choices'):
                return "Failed to generate description."
                
            # Get the generated description
            description = completion['choices'][0]['message']['content']
            return description
            
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return "Error generating description."

    async def analyze_query(self, query: str) -> Dict:
        """Simplified version that skips the classifier model analysis."""
        logger.info(f"[SIMPLE SEARCH] Skipping enhanced query analysis for: {shorten(query, width=100, placeholder='...')}")
        return {"success": False}  # Return a simple failure result to trigger fallback

    async def enhanced_search(self, query: str, analysis: Dict, model: str = None) -> List[Tuple[str, str, float, Optional[str], int, int]]:
        """Simplified version that directly uses document_manager.search."""
        logger.info(f"[SIMPLE SEARCH] Using basic search for: {shorten(query, width=100, placeholder='...')}")
        
        # Determine top_k based on model
        if model:
            top_k = self.config.get_top_k_for_model(model)
        else:
            top_k = self.config.TOP_K
            
        # Use document_manager.search directly
        return await self.document_manager.search(query, top_k=top_k) # Await async call
                
    async def split_query_into_sections(self, query: str) -> List[str]:
        """Simplified version that doesn't split the query."""
        logger.info(f"[SIMPLE SEARCH] Not splitting query into sections")
        return [query]  # Return the query as a single section

    async def process_multi_section_query(self, query: str, preferred_model: str = None) -> Dict:
        """Simplified version that skips section splitting and uses simple search."""
        logger.info(f"[SIMPLE SEARCH] Using simplified query processing for: {shorten(query, width=100, placeholder='...')}")
        
        # Simple analysis (will trigger fallback to simple search)
        analysis = {"success": False}
        
        # Simple search
        if preferred_model:
            top_k = self.config.get_top_k_for_model(preferred_model)
        else:
            top_k = self.config.TOP_K
            
        search_results = await self.document_manager.search(query, top_k=top_k) # Await async call
        
        # No synthesis
        synthesis = ""
        
        return {
            "search_results": search_results,
            "synthesis": synthesis,
            "analysis": analysis,
            "sections": [query]  # Just the original query
        }

    async def synthesize_results(self, query: str, search_results: List[Tuple[str, str, float, Optional[str]]], analysis: Dict) -> str:
        """Simplified version that returns an empty synthesis."""
        logger.info(f"[SIMPLE SEARCH] Skipping result synthesis for: {shorten(query, width=100, placeholder='...')}")
        return ""  # Return empty synthesis

    async def send_split_message(self, channel, text, reference=None, mention_author=False, model_used=None, user_id=None, existing_message=None):
        """Send a message split into chunks if it's too long, with each chunk referencing the previous one.
        Includes improved error handling, rate limiting awareness, and recovery mechanisms."""
        # Split the text into chunks
        chunks = split_message(text)
        
        if model_used and user_id:
            debug_mode = self.user_preferences_manager.get_debug_mode(user_id)
            if debug_mode:
                # Format the debug info to show the actual model used
                debug_info = f"\n\n*[Debug: Response generated using {model_used}]*"
                
                # Check if adding debug info would exceed the character limit
                if len(chunks[-1]) + len(debug_info) > 1750:
                    # Create a new chunk for the debug info
                    chunks.append(debug_info)
                else:
                    chunks[-1] += debug_info
        
        # Keep track of the last message sent to use as reference for the next chunk
        last_message = None
        failed_chunks = []
        
        # File fallback for very long responses
        if len(chunks) > 5:  # If response would be more than 4 messages
            try:
                # Create a temporary file with the full response
                file_content = text
                if model_used and user_id and debug_mode:
                    file_content += f"\n\n[Debug: Response generated using {model_used}]"
                    
                file_obj = io.StringIO(file_content)
                file = discord.File(file_obj, filename="publicia_response.txt")
                    
                # Send the file with a brief explanation
                await channel.send(
                    content="*neural pathways extended!* My response is quite long, so I've attached it as a file for easier reading.",
                    file=file,
                    reference=reference,
                    mention_author=mention_author
                )
                file_obj.close()
                return  # Exit early if file was sent successfully
            except Exception as e:
                logger.error(f"Error sending response as file, falling back to chunks: {e}")
                # Continue with normal chunk sending if file upload fails
        
        # Update existing message with first chunk if provided
        if existing_message and chunks:
            try:
                await existing_message.edit(content=chunks[0])
                last_message = existing_message
                chunks = chunks[1:]  # Remove the first chunk since it's already sent
            except discord.errors.NotFound:
                # Message was deleted, send as a new message
                logger.warning("Existing message not found, sending as new message")
                last_message = None
            except Exception as e:
                logger.error(f"Error editing existing message: {e}")
                last_message = None
        
        # For the first chunk (if no existing_message was provided or editing failed), use the original reference
        if chunks and last_message is None:
            max_retries = 3
            for retry in range(max_retries):
                try:
                    first_message = await channel.send(
                        content=chunks[0],
                        reference=reference,
                        mention_author=mention_author
                    )
                    last_message = first_message
                    chunks = chunks[1:]  # Remove the first chunk since it's already sent
                    break
                except Exception as e:
                    logger.error(f"Error sending first message chunk (attempt {retry+1}/{max_retries}): {e}")
                    await asyncio.sleep(1)  # Wait before retrying
            
            if last_message is None and chunks:
                # If we still couldn't send the first chunk after retries
                try:
                    await channel.send(
                        content="*neural circuit error* I'm having trouble sending my full response. Please try again later.",
                        reference=reference,
                        mention_author=mention_author
                    )
                except:
                    pass  # If even the error notification fails, just continue
        
        # Send remaining chunks sequentially, with retries and rate limit handling
        for i, chunk in enumerate(chunks):
            # Add continuation marker for non-first chunks
            if i > 0 or not chunks[0].startswith("*continued"):
                if not chunk.startswith("*continued") and not chunk.startswith("*code block"):
                    chunk = f"-# *continued response (part {i+2})*\n\n{chunk}"
            
            # Try to send the chunk with retry logic
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second delay
            success = False
            
            for retry in range(max_retries):
                try:
                    # Add a small delay before sending to avoid rate limits
                    await asyncio.sleep(retry_delay)
                    
                    # Each new chunk references the previous one to maintain the chain
                    if last_message:
                        new_message = await channel.send(
                            content=chunk,
                            reference=last_message,  # Reference the previous message in the chain
                            mention_author=False  # Don't mention for follow-up chunks
                        )
                    else:
                        # Fallback if we don't have a previous message to reference
                        new_message = await channel.send(
                            content=chunk,
                            reference=reference,
                            mention_author=False
                        )
                    
                    # Update reference for the next chunk
                    last_message = new_message
                    success = True
                    break  # Success, exit retry loop
                    
                except discord.errors.HTTPException as e:
                    # Check if it's a rate limit error
                    if e.status == 429:  # Rate limited
                        retry_after = float(e.response.headers.get('X-RateLimit-Reset-After', retry_delay * 2))
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        retry_delay = min(retry_delay * 2, 5)  # Exponential backoff with 5s cap
                    else:
                        # Other HTTP error, retry with backoff
                        logger.error(f"HTTP error sending chunk {i+2}: {e}")
                        retry_delay = min(retry_delay * 2, 5)
                        await asyncio.sleep(retry_delay)
                except Exception as e:
                    # General error, retry with backoff
                    logger.error(f"Error sending chunk {i+2}: {e}")
                    retry_delay = min(retry_delay * 2, 5)
                    await asyncio.sleep(retry_delay)
            
            if not success:
                # If all retries failed, add to failed chunks
                failed_chunks.append(chunk)
                logger.error(f"Failed to send chunk {i+2} after {max_retries} retries")
        
        # If any chunks failed to send, notify the user
        if failed_chunks:
            try:
                # Try to send failed chunks as a file
                missing_content = "\n\n".join(failed_chunks)
                file_obj = io.StringIO(missing_content)
                file = discord.File(file_obj, filename="missing_response.txt")
                
                await channel.send(
                    content=f"*neural circuit partially restored!* {len(failed_chunks)} parts of my response failed to send. I've attached the missing content as a file.",
                    file=file,
                    reference=last_message or reference,
                    mention_author=False
                )
                file_obj.close()
            except Exception as e:
                logger.error(f"Error sending missing chunks as file: {e}")
                
                # If file upload fails, try to send a simple notification
                try:
                    await channel.send(
                        content=f"*neural circuit overload!* {len(failed_chunks)} parts of my response failed to send. Please try asking again later.",
                        reference=last_message or reference,
                        mention_author=False
                    )
                except:
                    pass  # If even this fails, give up

    async def refresh_google_docs(self):
        """Refresh all tracked Google Docs."""
        tracked_file = Path(self.document_manager.base_dir) / "tracked_google_docs.json"
        if not tracked_file.exists():
            return
            
        try:
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
        
            updated_docs = False  # Track if any docs were updated
            
            for doc in tracked_docs:
                try:
                    doc_id = doc['id']
                    # Determine original and sanitized names
                    original_name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                    if not original_name.endswith('.txt'):
                        original_name += '.txt'
                    sanitized_name = self.document_manager._sanitize_name(original_name)
                    logger.info(f"Refresh: Processing Google Doc ID {doc_id}: Original='{original_name}', Sanitized='{sanitized_name}'")

                    # Check if document has changed using sanitized name for metadata lookup
                    changed = await self._has_google_doc_changed(doc_id, sanitized_name)
                    
                    if changed:
                        logger.info(f"Google Doc {doc_id} has changed, updating")
                        
                        async with aiohttp.ClientSession() as session:
                            url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                            async with session.get(url) as response:
                                if response.status != 200:
                                    logger.error(f"Failed to download {doc_id}: {response.status}")
                                    continue
                                content = await response.text()

                        # Save to file using SANITIZED name
                        file_path = self.document_manager.base_dir / sanitized_name
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Refresh: Saved Google Doc {doc_id} content to file: '{file_path}'")
                        except Exception as write_err:
                            logger.error(f"Refresh: Failed to write Google Doc content to '{file_path}': {write_err}", exc_info=True)
                            continue # Skip to next doc if write fails

                        # Add/Update document in manager using ORIGINAL name
                        logger.info(f"Refresh: Adding/Updating document in manager with original name: '{original_name}'")
                        # Use save_to_disk=False to batch the final save
                        add_success = await self.document_manager.add_document(original_name, content, save_to_disk=False)
                        if add_success:
                            updated_docs = True
                        else:
                             logger.error(f"Refresh: Document manager failed to add/update document for '{original_name}'")
                    else:
                        logger.info(f"Refresh: Google Doc {doc_id} ('{original_name}') has not changed, skipping")
                        
                except Exception as e:
                    logger.error(f"Error refreshing doc {doc_id}: {e}")
            
            # Save to disk once at the end if any docs were updated
            if updated_docs:
                self.document_manager._save_to_disk()
        except Exception as e:
            logger.error(f"Error refreshing Google Docs: {e}")

    async def _has_google_doc_changed(self, doc_id: str, file_name: str) -> bool:
        """Check if a Google Doc has changed since last refresh using content hashing
           against the hash stored in the main DocumentManager metadata."""
        try:
            # Get the stored hash from the main DocumentManager metadata
            stored_hash = self.document_manager.metadata.get(file_name, {}).get('content_hash')

            # Download the latest version from Google
            async with aiohttp.ClientSession() as session:
                url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to download {doc_id} for change check (Status: {response.status}). Assuming changed.")
                        return True # Assume changed if download fails
                    new_content = await response.text()

            # Compute hash of the newly downloaded content
            import hashlib
            new_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()

            # Compare the new hash with the stored hash
            if stored_hash and stored_hash == new_hash:
                # Hashes match, document has not changed
                logger.debug(f"Google Doc {doc_id} ({file_name}) hash matched ({new_hash[:8]}...). No change detected.")
                return False
            else:
                # Hashes differ or no stored hash exists, document has changed
                if stored_hash:
                    logger.info(f"Google Doc {doc_id} ({file_name}) hash mismatch. Stored: {stored_hash[:8]}..., New: {new_hash[:8]}.... Change detected.")
                else:
                    logger.info(f"Google Doc {doc_id} ({file_name}) has no stored hash. Assuming changed.")
                return True

        except Exception as e:
            logger.error(f"Error checking if Google Doc {doc_id} ({file_name}) has changed: {e}")
            # If there's any error during the check, assume it has changed to be safe
            return True
            
    async def refresh_single_google_doc(self, doc_id: str, custom_name: str = None) -> bool:
        """Refresh a single Google Doc by its ID."""
        try:
            # Get the current mapping to check for name changes
            doc_mapping = self.document_manager.get_googledoc_id_mapping()
            old_filename = None
            
            # Find if this doc_id exists with a different filename
            for filename, mapped_id in doc_mapping.items():
                if mapped_id == doc_id and filename != (custom_name or f"googledoc_{doc_id}.txt"):
                    old_filename = filename
                    break
            
            # Determine the intended original file name
            original_name = custom_name or f"googledoc_{doc_id}.txt"
            if not original_name.endswith('.txt'):
                original_name += '.txt'

            # Determine the sanitized name for file system operations
            sanitized_name = self.document_manager._sanitize_name(original_name)
            logger.info(f"Processing Google Doc ID {doc_id}: Original Name='{original_name}', Sanitized Name='{sanitized_name}'")

            # Check if document has changed using the sanitized name for metadata lookup
            changed = await self._has_google_doc_changed(doc_id, sanitized_name)
            
            if not changed:
                logger.info(f"Google Doc {doc_id} has not changed, skipping")
                return True  # Return success, but no update needed
            
            # Download the document
            async with aiohttp.ClientSession() as session:
                url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {doc_id}: {response.status}")
                        return False
                    content = await response.text()
            
            # Save to file using the SANITIZED name
            file_path = self.document_manager.base_dir / sanitized_name
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Saved Google Doc {doc_id} content to file: '{file_path}'")
            except Exception as write_err:
                 logger.error(f"Failed to write Google Doc content to '{file_path}': {write_err}", exc_info=True)
                 return False # Abort if file write fails

            # If the custom name changed, we might need to remove data associated with the OLD sanitized name
            # Note: old_filename here is the *original* name from the mapping before the change
            if old_filename:
                 old_sanitized_name = self.document_manager._sanitize_name(old_filename)
                 if old_sanitized_name != sanitized_name and old_sanitized_name in self.document_manager.metadata:
                     logger.info(f"Removing old document data for previous name '{old_filename}' (sanitized: {old_sanitized_name})")
                     # Use delete_document with the OLD ORIGINAL name to handle removal cleanly
                     await self.document_manager.delete_document(old_filename) # Let delete handle file removal too

            # Add/Update document in the manager using the ORIGINAL name
            # add_document will handle internal sanitization and metadata correctly
            logger.info(f"Adding/Updating document in manager with original name: '{original_name}'")
            success = await self.document_manager.add_document(original_name, content) # Let add_document handle saving

            if not success:
                 logger.error(f"Document manager failed to add/update document for '{original_name}'")
                 # Clean up the file we just saved if add failed? Maybe not necessary if add_document handles errors well.
                 return False

            return True
            
            return True
                
        except Exception as e:
            logger.error(f"Error downloading doc {doc_id}: {e}")
            return False

    async def setup_hook(self):
        """Initial setup hook called by discord.py."""
        logger.info("Bot is setting up...")
        
        # Load documents asynchronously
        await self.document_manager._load_documents()
        
        # Setup commands and sync
        await self.setup_commands()
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def setup_commands(self):
        """Set up all slash and prefix commands from separate modules."""
        from commands import (
            document_commands,
            image_commands,
            conversation_commands,
            admin_commands,
            utility_commands,
            query_commands
        )

        # Register all commands
        document_commands.register_commands(self)
        image_commands.register_commands(self)
        conversation_commands.register_commands(self)
        admin_commands.register_commands(self)
        utility_commands.register_commands(self)
        query_commands.register_commands(self)

        logger.info("All commands registered successfully")

    async def on_ready(self):
        sys.stdout.flush()  # Explicitly flush stdout before logging
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        
    async def _extract_google_doc_ids(self, text: str) -> List[Tuple[str, str]]:
        """Extract Google Doc IDs from text."""
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
        """Fetch the content of a Google Doc without tracking it."""
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
        """Fetch the title of a Google Doc."""
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
                        title = re.sub(r'\s*-\s*Google\s*Docs$', '', title)
                        return title
                    
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error getting title for doc {doc_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting title for doc {doc_id}: {e}")
            return None
            
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
    
    async def _try_ai_completion(self, model: Union[str, List[str]], messages: List[Dict], image_ids=None, image_attachments=None, temperature=0.1, max_retries=2, min_response_length=5, **kwargs) -> Tuple[Optional[Any], Optional[str]]:
        """Get AI completion with dynamic fallback options or a specific model list.

        Args:
            model (str | List[str]): The primary model ID (str) to use with automatic fallbacks,
                                     or an explicit list of model IDs (List[str]) to try in order.
            messages (List[Dict]): The message history for the API call.
            image_ids (Optional[List[str]]): List of image IDs from search results.
            image_attachments (Optional[List[str]]): List of base64 encoded image attachments.
            temperature (float): The sampling temperature.
            max_retries (int): Max retries for short responses on the *same* model.
            min_response_length (int): Minimum character length for a valid response.
            **kwargs: Additional arguments for the API call.

        Returns:
            Tuple[Optional[Dict], Optional[str]]: (completion result, actual model used)
        """
        models_to_try = []
        requested_model_or_list = model # Keep track of original request for logging

        if isinstance(model, list):
            # If a list is provided, use it directly
            models_to_try = model
            logger.info(f"Using explicit model list for completion: {models_to_try}")
        elif isinstance(model, str):
            # If a string is provided, build the fallback list as before
            logger.info(f"Building fallback list starting with requested model: {model}")
            models_to_try = [model] # Start with the requested model
            model_family = model.split('/')[0] if '/' in model else None

            # Add model-specific fallbacks
            # (Keep the existing fallback logic here)
            # DeepSeek models
            if "deepseek/deepseek-chat-v3" in model:
                # DeepSeek Chat v3 fallbacks
                fallbacks = [
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-chat-v3-0324:floor",
                    "deepseek/deepseek-chat-v3-0324",
                    "deepseek/deepseek-chat",
                    "deepseek/deepseek-r1:free",  # Last resort fallback to R1
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            elif "deepseek/deepseek-r1" in model:
                # DeepSeek R1 fallbacks
                fallbacks = [
                    "deepseek/deepseek-r1:free",
                    "deepseek/deepseek-r1:floor",
                    "deepseek/deepseek-r1",
                    "deepseek/deepseek-r1:nitro",
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "deepseek/deepseek-r1-distill-qwen-32b"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            elif "deepseek/deepseek-chat" in model and "v3" not in model:
                # DeepSeek Chat (non-v3) fallbacks
                fallbacks = [
                    "deepseek/deepseek-chat",
                    "deepseek/deepseek-r1:free",
                    "deepseek/deepseek-r1"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Grok
            elif "grok" in model:
                fallbacks = [
                    "x-ai/grok-3-mini-beta",
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Llama
            elif "meta-llama/llama-4-maverick" in model:
                fallbacks = [
                    "meta-llama/llama-4-maverick:floor",
                    "meta-llama/llama-4-maverick",
                    "meta-llama/llama-4-scout",
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Qwen models
            elif model_family == "qwen":
                fallbacks = [
                    "qwen/qwq-32b:free",
                    "qwen/qwq-32b:floor",
                    "qwen/qwq-32b",
                    "qwen/qwen-turbo",
                    "qwen/qwen2.5-32b-instruct"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Google models
            elif model_family == "google":
                fallbacks = [
                    "google/gemini-2.5-flash-preview",
                    "google/gemini-2.5-pro-exp-03-25:free", # Add new model as a primary fallback
                    "google/gemini-2.0-flash-thinking-exp:free",
                    "google/gemini-2.0-pro-exp-02-05:free",
                    "google/gemini-2.0-flash-001"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # TheDrummer models
            elif model_family == "thedrummer" and "anubis" in model:
                fallbacks = [
                    "thedrummer/anubis-pro-105b-v1",
                    "latitudegames/wayfarer-large-70b-llama-3.3",  # try other narrative model
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            elif model_family == "thedrummer":  # Other TheDrummer models
                fallbacks = [
                    "thedrummer/unslopnemo-12b",
                    "thedrummer/rocinante-12b",
                    "meta-llama/llama-3.3-70b-instruct"  # Safe fallback option
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Eva models
            elif model_family == "eva-unit-01":
                fallbacks = [
                    "eva-unit-01/eva-qwen-2.5-72b:floor",
                    "eva-unit-01/eva-qwen-2.5-72b",
                    "qwen/qwq-32b:free",
                    "qwen/qwq-32b",
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Nous Research models
            elif model_family == "nousresearch":
                fallbacks = [
                    "nousresearch/hermes-3-llama-3.1-70b",
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "meta-llama/llama-3.3-70b-instruct"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Anthropic models
            elif model_family == "anthropic":
                if "claude-3.7-sonnet" in model:
                    fallbacks = [
                        "anthropic/claude-3.7-sonnet",
                        "anthropic/claude-3.5-sonnet:beta",
                        "anthropic/claude-3.5-haiku:beta",
                        "anthropic/claude-3.5-haiku"
                    ]
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
                elif "claude-3.5-sonnet" in model:
                    fallbacks = [
                        "anthropic/claude-3.5-sonnet",
                        "anthropic/claude-3.7-sonnet:beta",
                        "anthropic/claude-3.7-sonnet",
                        "anthropic/claude-3.5-haiku:beta",
                        "anthropic/claude-3.5-haiku"
                    ]
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
                else:
                    # General Anthropic fallbacks
                    fallbacks = [
                        "latitudegames/wayfarer-large-70b-llama-3.3",
                        "meta-llama/llama-3.3-70b-instruct",  # base model fallback
                    ]
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # Microsoft models
            elif model_family == "microsoft":
                fallbacks = [
                    "microsoft/phi-4-multimodal-instruct",
                    "microsoft/phi-4",
                    "microsoft/phi-3.5-mini-128k-instruct"
                ]
                models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
            # OpenAI
            elif model_family == "openai":
                if "4.1-mini" in model:
                    fallbacks = [
                        "openai/gpt-4.1-mini",
                    ]
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
                elif "4.1-nano" in model:
                    fallbacks = [
                        "openai/gpt-4.1-nano",
                    ]
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])
                else:
                    fallbacks = [] # Ensure fallbacks is defined even if no specific match
                    models_to_try.extend([fb for fb in fallbacks if fb not in models_to_try])

            # Add general fallbacks (cleaned up)
            general_fallbacks = [
                "google/gemini-2.5-pro-exp-03-25:free",
                "google/gemini-2.5-flash-preview",
                "qwen/qwq-32b:free",
                "qwen/qwq-32b:floor",
                "google/gemini-2.5-flash-preview",
                "google/gemini-2.0-flash-thinking-exp:free",
                "deepseek/deepseek-r1:free",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-chat",
                "google/gemini-2.0-pro-exp-02-05:free",
                "nousresearch/hermes-3-llama-3.1-405b",
                "anthropic/claude-3.5-haiku:beta",
                "anthropic/claude-3.5-haiku"
            ]
            models_to_try.extend([fb for fb in general_fallbacks if fb not in models_to_try])
            logger.info(f"Final fallback list: {models_to_try}")
        else:
             logger.error(f"Invalid type for 'model' parameter: {type(model)}. Expected str or List[str].")
             return None, None # Cannot proceed

        # Check if we need a vision-capable model
        need_vision = (image_ids and len(image_ids) > 0) or (image_attachments and len(image_attachments) > 0)

        # Headers for API calls
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "Publicia - DPS",
            "Content-Type": "application/json"
        }

        for current_model in models_to_try:
            try:
                logger.info(f"Attempting completion with model: {current_model}")
                
                # Check if current model supports vision
                is_vision_model = current_model in self.vision_capable_models
                
                # Prepare messages based on whether we're using a vision model
                processed_messages = messages.copy()
                
                # If we have images and this is a vision-capable model, add them to the last user message
                if need_vision and is_vision_model:
                    # Find the last user message
                    for i in range(len(processed_messages) - 1, -1, -1):
                        if processed_messages[i]["role"] == "user":
                            # Convert the content to the multimodal format
                            user_msg = processed_messages[i]
                            text_content = user_msg["content"]
                            
                            # Create a multimodal content array
                            content_array = [{"type": "text", "text": text_content}]
                            
                            # Add each image from attachments
                            if image_attachments:
                                for img_data in image_attachments:
                                    if img_data:  # Only add if we have valid image data
                                        content_array.append({
                                            "type": "image_url",
                                            "image_url": {"url": img_data}
                                        })
                                        logger.info(f"Added direct attachment image to message")
                            
                            # Add each image from image_ids
                            if image_ids:
                                for img_id in image_ids:
                                    try:
                                        # Get base64 image data
                                        base64_image = self.image_manager.get_base64_image(img_id)
                                        content_array.append({
                                            "type": "image_url",
                                            "image_url": {"url": base64_image}
                                        })
                                        logger.info(f"Added search result image {img_id} to message")
                                    except Exception as e:
                                        logger.error(f"Error adding image {img_id} to message: {e}")
                            
                            # Replace the content with the multimodal array
                            processed_messages[i]["content"] = content_array
                            
                            # Log the number of images added
                            image_count = len(content_array) - 1  # Subtract 1 for the text content
                            logger.info(f"Added {image_count} images to message for vision model")
                            break

                provider_config = self.config.get_provider_config(current_model)
                
                payload = {
                    "model": current_model,
                    "messages": processed_messages,
                    "temperature": temperature,
                    "max_tokens": 20000,
                    **kwargs
                }

                if provider_config:
                    payload["provider"] = provider_config
                    logger.info(f"Using custom provider configuration for {current_model}: {provider_config}")

                if current_model.startswith("deepseek/"):
                    payload["max_price"] = {
                        "completion": "4",
                        "prompt": "2"
                    }
                    logger.info(f"Adding max_price parameter for DeepSeek model {current_model}: completion=4, prompt=2")
                
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
                                logger.error(f"API error (Status {response.status}): {error_text}")
                                # Log additional context like headers to help diagnose issues
                                logger.error(f"Request context: URL={response.url}, Headers={response.headers}")
                                return None
                                
                            return await response.json()

                completion = await asyncio.wait_for(
                    api_call(),
                    timeout=self.timeout_duration
                )
                
                if completion and completion.get('choices') and len(completion['choices']) > 0:
                    if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                        response_content = completion['choices'][0]['message']['content']
                        
                        # Check if response is too short (implement retry logic)
                        if len(response_content.strip()) < min_response_length:
                            logger.warning(f"Response from {current_model} is too short ({len(response_content.strip())} chars): '{response_content}'")
                            
                            # If we have retries left, try again with the same model (possibly with higher temperature)
                            if kwargs.get('_retry_count', 0) < max_retries:
                                logger.info(f"Retrying with {current_model} (retry {kwargs.get('_retry_count', 0) + 1}/{max_retries})")
                                
                                # Create a copy of kwargs with incremented retry count and slightly higher temperature
                                retry_kwargs = kwargs.copy()
                                retry_kwargs['_retry_count'] = kwargs.get('_retry_count', 0) + 1
                                
                                # Increase temperature slightly for retry (but cap it)
                                retry_temp = min(temperature * 1.2, 0.9)  # Increase by 20% but max 0.9
                                
                                # Recursive call to retry with the same model
                                return await self._try_ai_completion(
                                    current_model, 
                                    messages, 
                                    image_ids, 
                                    image_attachments, 
                                    retry_temp, 
                                    max_retries,
                                    min_response_length,
                                    **retry_kwargs
                                )
                            else:
                                # If we've used all retries for this model, log it and continue to the next model
                                logger.warning(f"Used all retries for {current_model}, continuing to next model")
                                continue
                        
                        # Normal case - response is long enough
                        logger.info(f"Successful completion from {current_model}")
                        logger.info(f"Response: {shorten(response_content, width=200, placeholder='...')}")
                        
                        # For analytics, log which model was actually used
                        if requested_model_or_list != current_model and isinstance(requested_model_or_list, str):
                            logger.info(f"Notice: Fallback model {current_model} was used instead of requested {requested_model_or_list}")
                        elif isinstance(requested_model_or_list, list) and current_model != requested_model_or_list[0]:
                             logger.info(f"Notice: Model {current_model} was used from the provided list (requested first: {requested_model_or_list[0]})")

                        return completion, current_model  # Return both the completion and the model used
                    else:
                        logger.error(f"Unexpected response structure from {current_model}: {completion}")
                        # Return the incomplete response anyway - let the caller handle it
                        return completion, current_model
                    
            except Exception as e:
                # Get the full traceback information
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Error with model {current_model}: {str(e)}\nTraceback:\n{tb}")
                continue
        
        logger.error(f"All models failed to generate completion. Attempted models: {', '.join(models_to_try)}")
        return None, None  # Return None for both completion and model used

    def calculate_dynamic_temperature(self, query: str, conversation_history=None):
        """
        Calculates appropriate temperature based on query type:
        - Lower (TEMPERATURE_MIN-TEMPERATURE_BASE) for factual/information queries 
        - Higher (TEMPERATURE_BASE-TEMPERATURE_MAX) for creative/roleplay scenarios
        - Base of TEMPERATURE_BASE for balanced queries
        """
        # Get temperature constants from config
        BASE_TEMP = self.config.TEMPERATURE_BASE
        MIN_TEMP = self.config.TEMPERATURE_MIN
        MAX_TEMP = self.config.TEMPERATURE_MAX
        
        # Normalize query
        query = query.lower().strip()
        
        # Score tracking
        roleplay_score = 0.0
        information_score = 0.0
        
        # === ROLEPLAY DETECTION ===
        
        # Action descriptions with asterisks
        action_count = len(re.findall(r"\*[^*]+\*", query))
        if action_count > 0:
            roleplay_score += min(1.5, action_count * 0.5)
        
        # Dialogue markers
        if re.search(r"[\"'].+?[\"']", query):
            roleplay_score += 0.8
        
        # Roleplay phrases
        roleplay_phrases = [
            # Basic roleplay indicators
            "roleplay", "in character", "act as", "conversation", "scene", "scenario",
            
            # Speech indicators
            "says", "say", "speak", "speaks", "said", "speaking", "talk", "talks", 
            "reply", "replies", "respond", "responds", "answered", "tells", "told",
            
            # Action verbs
            "does", "do", "perform", "performs", "acted", "acting", "moves", "moved",
            "walks", "sits", "stands", "turns", "looks", "smiles", "frowns", "nods",
            
            # Narrative elements
            "narrate", "describe scene", "setting", "environment", "continues", 
            "starts", "begins", "enters", "exits", "appears", "suddenly",
            
            # Character emotions/states
            "feeling", "felt", "emotion", "expression", "mood", "attitude",
            "surprised", "excited", "nervous", "calm", "angry"
        ]
        if any(phrase in query for phrase in roleplay_phrases):
            roleplay_score += 1.5
            
        # First-person narrative (common in roleplay)
        first_person_count = len(re.findall(r"\b(i|me|my|mine|myself)\b", query))
        if first_person_count > 1:
            roleplay_score += 0.5
        
        # === INFORMATION DETECTION ===
        
        # Question indicators
        question_markers = ["?", "what", "who", "where", "when", "why", "how"]
        if any(marker in query.split() for marker in question_markers) or "?" in query:
            information_score += 0.6
        
        # Information-seeking phrases
        info_phrases = ["explain", "describe", "tell me", "information", "info", "details"]
        if any(phrase in query for phrase in info_phrases):
            information_score += 1.2
        
        # Lore-specific terms
        lore_terms = ["ledus banum", "tundra", "empire", "imperial", "lore", 
                    "history", "institution", "house", "region", "church"]
        if any(term in query for term in lore_terms):
            information_score += 1.0
        
        # === CONVERSATION CONTEXT ===
        
        # Check previous messages for roleplay context
        if conversation_history and len(conversation_history) > 0:
            recent_msgs = conversation_history[-min(3, len(conversation_history)):]
            
            for msg in recent_msgs:
                msg_content = msg.get("content", "").lower()
                if "*" in msg_content or any(phrase in msg_content for phrase in roleplay_phrases):
                    roleplay_score += 0.8
                    break
        
        # === CALCULATE FINAL TEMPERATURE ===
        
        # If both scores are very low, use base temperature
        if roleplay_score < 0.5 and information_score < 0.5:
            return BASE_TEMP
        
        # Calculate ratio of roleplay vs information
        total_score = roleplay_score + information_score
        if total_score > 0:
            roleplay_ratio = roleplay_score / total_score
        else:
            roleplay_ratio = 0.5
        
        # Map ratio to temperature range
        temp_range = MAX_TEMP - MIN_TEMP
        temperature = MIN_TEMP + (roleplay_ratio * temp_range)
        
        # Ensure we're within boundaries
        temperature = max(MIN_TEMP, min(MAX_TEMP, temperature))
        
        # Log for debugging
        logger.info(f"Query temp analysis: '{query[:30]}...' - Roleplay: {roleplay_score:.1f}, Info: {information_score:.1f}, Temp: {temperature:.2f} [Range: {MIN_TEMP}-{MAX_TEMP}]")        
        
        return temperature

    def is_context_dependent_query(self, query: str) -> bool:
        """Determine if a query likely depends on conversation context."""
        query = query.lower().strip()
        
        # 1. Very short queries are suspicious (2-3 words)
        if 1 <= len(query.split()) <= 3:
            return True
            
        # 2. Queries with pronouns suggesting reference to previous content
        pronouns = ["they", "them", "these", "those", "this", "that", "it", "he", "she",
                    "their", "its", "his", "her"]
        if any(f" {p} " in f" {query} " for p in pronouns):
            return True
            
        # 3. Queries explicitly asking for more/additional information
        continuation_phrases = ["more", "another", "additional", "else", "other", 
                               "elaborate", "continue", "expand", "also", "further",
                               "example", "examples", "specifically", "details"]
        if any(phrase in query.split() for phrase in continuation_phrases):
            return True
            
        # 4. Queries starting with comparison words
        if re.match(r"^(what about|how about|compared to|similarly|unlike|like)", query):
            return True
            
        # 5. Incomplete-seeming questions
        if re.match(r"^(and|but|so|or|then|why not|why|how)\b", query):
            return True
        
        return False

    def get_conversation_context(self, username: str, current_query: str) -> str:
        """Extract relevant context from conversation history."""
        # Get recent messages
        conversation = self.conversation_manager.get_conversation_messages(username, limit=6)
        
        if len(conversation) <= 1:
            return ""
        
        # Extract the last substantive user query before this one
        prev_user_query = ""
        for msg in reversed(conversation[:-1]):  # Skip the current query
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # A substantive query is reasonably long and not itself context-dependent
                if len(content.split()) > 3 and not self.is_context_dependent_query(content):
                    prev_user_query = content
                    break
        
        # Extract important entities/topics from the last assistant response
        last_assistant_response = ""
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                last_assistant_response = msg.get("content", "")
                break
                
        # If we have a previous query, use that as primary context
        if prev_user_query:
            # Extract the main subject by removing question words and common articles
            query_words = prev_user_query.lower().split()
            question_words = ["what", "who", "where", "when", "why", "how", "is", "are", 
                             "was", "were", "will", "would", "can", "could", "do", "does",
                             "did", "has", "have", "had", "tell", "me", "about", "please"]
            
            # Keep only the first 6-8 content words
            content_words = [w for w in query_words if w not in question_words][:8]
            context = " ".join(content_words) if content_words else prev_user_query
            
            return context
            
        # Fallback: if no good previous query, try to extract nouns/subjects from last response
        if last_assistant_response:
            # Very basic approach: look for capitalized words that might be important entities
            sentences = last_assistant_response.split('.')
            for sentence in sentences[:3]:  # Check first few sentences
                words = sentence.split()
                proper_nouns = [word for word in words 
                              if word and word[0].isupper() and len(word) > 1]
                if proper_nouns:
                    return " ".join(proper_nouns[:5])
        
        return ""

    # Note: This function exists but is not actively used in the primary context-aware search flow,
    # which now relies on generate_context_aware_embedding.
    def enhance_context_dependent_query(self, query: str, context: str) -> str:
        """(Legacy/Unused) Enhance a context-dependent query text with conversation context string."""
        if not context:
            return query
            
        query = query.strip()
        context = context.strip()
        
        # 1. For very minimal queries like "more" or "continue"
        if query.lower() in ["more", "continue", "go on", "and", "then"]:
            return f"Tell me more about {context}"
            
        # 2. For queries asking for examples
        if re.match(r"^examples?(\s|$|\?)", query.lower()):
            return f"Give examples of {context}"
            
        # 3. For "what about X" queries
        if re.match(r"^what about|how about", query.lower()):
            remaining = re.sub(r"^what about|^how about", "", query.lower()).strip()
            return f"What about {remaining} in relation to {context}"
            
        # 4. For queries starting with pronouns
        for pronoun in ["they", "them", "these", "those", "this", "that", "it"]:
            if re.match(f"^{pronoun}\\b", query.lower()):
                # Replace the pronoun with the context
                return re.sub(f"^{pronoun}\\b", context, query, flags=re.IGNORECASE)
        
        # 5. Default approach: explicitly add context
        if query.endswith("?"):
            # Add context parenthetically for questions
            return f"{query} (regarding {context})"
        else:
            # Add context with "about" or "regarding"
            return f"{query} about {context}"

    def cache_search_results(self, username: str, query: str, results):
        """Store search results for potential follow-ups."""
        # Only cache if we have decent results
        if not results or len(results) < 2:
            return
            
        self.search_cache[username] = {
            'query': query,
            'results': results,
            'used_indices': set(range(min(5, len(results)))),  # Track which results were already shown
            'timestamp': datetime.now()
        }
        logger.info(f"Cached {len(results)} search results for {username}, initially showed {len(self.search_cache[username]['used_indices'])}")
    
    def get_additional_results(self, username: str, top_k=3):
        """Get additional unseen results from previous search."""
        if username not in self.search_cache:
            return []
            
        cache = self.search_cache[username]
        
        # Check if cache is too old (5 minutes)
        if (datetime.now() - cache['timestamp']).total_seconds() > 300:
            logger.info(f"Cache for {username} expired, ignoring")
            return []
        
        # Find results not yet shown
        new_results = []
        for i, result in enumerate(cache['results']):
            if i not in cache['used_indices'] and len(new_results) < top_k:
                new_results.append(result)
                cache['used_indices'].add(i)
        
        if new_results:
            logger.info(f"Found {len(new_results)} additional unused results for {username}")
        
        return new_results

    async def generate_context_aware_embedding(self, query: str, conversation_history: List[Dict]): # Accept history list
        """Generate an embedding that combines current query with weighted conversation history."""

        # 1. Select relevant history (e.g., last 5 messages, excluding the current query if present)
        # Ensure history is ordered chronologically (oldest to newest)
        relevant_history = conversation_history[-6:-1] # Get up to 5 messages before the last one (which is often the current query context)
        if not relevant_history:
            # No history, use normal query embedding
            embedding_result = await self.document_manager.generate_embeddings([query], is_query=True)
            if embedding_result.size == 0:
                 logger.error("Failed to generate query embedding (no history).")
                 return None
            return embedding_result[0]

        # 2. Prepare texts for embedding (query + history messages content)
        texts_to_embed = [query] + [msg.get("content", "") for msg in relevant_history]

        # 3. Generate embeddings for all texts at once
        embeddings = await self.document_manager.generate_embeddings(texts_to_embed, is_query=True)
        if embeddings.size == 0 or embeddings.shape[0] != len(texts_to_embed):
            logger.error("Failed to generate embeddings for context-aware search (history variants).")
            # Fallback to simple query embedding
            embedding_result = await self.document_manager.generate_embeddings([query], is_query=True)
            if embedding_result.size == 0: return None
            return embedding_result[0]

        query_embedding = embeddings[0]
        history_embeddings = embeddings[1:]

        # 4. Define weighting scheme (e.g., exponential decay)
        decay_factor = 0.8  # Newer messages get higher weight
        weights = np.array([decay_factor**i for i in range(len(history_embeddings) - 1, -1, -1)]) # Weights: [..., 0.8^2, 0.8^1, 0.8^0]
        if np.sum(weights) > 0: # Avoid division by zero if weights sum to 0
            weights /= np.sum(weights) # Normalize weights
        else:
            weights = np.ones(len(history_embeddings)) / len(history_embeddings) # Fallback to equal weights

        # 5. Calculate weighted average of history embeddings
        weighted_history_embedding = np.sum(history_embeddings * weights[:, np.newaxis], axis=0)

        # 6. Combine query embedding and weighted history embedding
        # Adjust weights as needed, e.g., 70% query, 30% history context
        combined_embedding = 0.7 * query_embedding + 0.3 * weighted_history_embedding

        # 7. Normalize the final embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            final_embedding = combined_embedding / norm
        else:
            # Fallback to query embedding if norm is zero
            final_embedding = query_embedding
            logger.warning("Combined embedding norm was zero, falling back to query embedding.")

        logger.info(f"Generated context-aware embedding using {len(relevant_history)} history messages.")
        return final_embedding

    async def process_hybrid_query(self, question: str, username: str, max_results: int = 5, use_context: bool = True): # Make async
        """Process queries using a hybrid of caching and context-aware embeddings with re-ranking."""
        # Skip context logic completely if use_context is False
        if not use_context:
            # Just do regular search with reranking asynchronously
            search_results = await self.document_manager.search( # Await async call
                question,
                top_k=max_results,
                apply_reranking=self.config.RERANKING_ENABLED
            )
            
            # Still cache results for consistency
            self.cache_search_results(username, question, search_results)
            return search_results # Added return statement here

        # --- Simplified Logic: Always attempt context-aware embedding ---
        # The generate_context_aware_embedding function handles the fallback
        # to standard query embedding if no relevant history exists.

        logger.info("Attempting context-aware search using conversation history (will fallback if no history)")

        # Get conversation history list
        # Fetch slightly more history as context for the embedding generation
        conversation_history = self.conversation_manager.get_conversation_messages(username, limit=10)

        # Generate context-aware embedding (or standard embedding if no history)
        embedding = await self.generate_context_aware_embedding(question, conversation_history)

        if embedding is None:
             logger.error("Failed to generate any embedding, falling back to basic keyword search (if implemented) or empty results.")
             # TODO: Implement a basic keyword search fallback?
             # For now, return empty results if embedding fails completely.
             return [] # Return empty list if embedding generation failed

        # Search with the generated embedding (context-aware or standard)
        apply_reranking = self.config.RERANKING_ENABLED

        if apply_reranking:
            # Get more initial results for re-ranking
            initial_results = self.document_manager.custom_search_with_embedding(
                embedding,
                top_k=self.config.RERANKING_CANDIDATES
            )

            # Apply re-ranking asynchronously
            if initial_results:
                logger.info(f"Applying re-ranking to {len(initial_results)} results")
                results = await self.document_manager.rerank_results(question, initial_results, top_k=max_results) # Await async call
            else:
                results = [] # No initial results to rerank
        else:
            # If re-ranking is disabled, perform direct search with the embedding
            results = self.document_manager.custom_search_with_embedding(embedding, top_k=max_results)

        # Cache results (optional, but kept for potential future use)
        # Note: The immediate utility for follow-ups is reduced as context is always checked now.
        self.cache_search_results(username, question, results)

        return results
        # --- End Simplified Logic ---

        # --- Old Logic (Commented out for reference) ---
        # is_followup = self.is_context_dependent_query(question)
        # original_question = question
        #
        # # # For standard non-follow-up queries
        # # if not is_followup:
        # #     # Determine whether to apply re-ranking
        # #     apply_reranking = self.config.RERANKING_ENABLED
        # #
        # #     # Do regular search with re-ranking asynchronously
        # #     search_results = await self.document_manager.search( # Await async call
        # #         question,
        # #         top_k=max_results,
        # #         apply_reranking=apply_reranking
        # #     )
        # #
        # #     # Cache for future follow-ups
        # #     self.cache_search_results(username, question, search_results)
        # #     return search_results
        # #
        # # # For follow-up queries
        # # logger.info(f"Detected follow-up query: '{question}'")
        # #
        # # # Try to get more results from previous search
        # # cached_results = self.get_additional_results(username, top_k=max_results)
        # #
        # # if cached_results:
        # #     # We have unused results, no need for new search
        # #     logger.info(f"Using {len(cached_results)} cached results")
        # #     return cached_results
        # #
        # # # No cached results, use context-aware search
        # # logger.info("No cached results, performing context-aware search using conversation history")
        # #
        # # # Get conversation history list
        # # conversation_history = self.conversation_manager.get_conversation_messages(username, limit=10) # Get more history for embedding
        # # if conversation_history:
        # #     logger.info(f"Using {len(conversation_history)} messages for context-aware embedding")
        # #     # Generate context-aware embedding using history list
        # #     embedding = await self.generate_context_aware_embedding(question, conversation_history) # Pass history list
        # #
        # #     if embedding is None:
        # #          logger.error("Failed to generate context-aware embedding, falling back to standard search.")
        # #          # Fallback to normal search with re-ranking asynchronously
        # #          search_results = await self.document_manager.search( # Await async call
        # #              question,
        # #              top_k=max_results,
        # #              apply_reranking=self.config.RERANKING_ENABLED
        # #          )
        # #          return search_results
        # #
        # #     # Search with this embedding and apply re-ranking
        # #     apply_reranking = self.config.RERANKING_ENABLED
        # #
        # #     if apply_reranking:
        # #         # Get more initial results for re-ranking
        # #         initial_results = self.document_manager.custom_search_with_embedding(
        # #             embedding,
        # #             top_k=self.config.RERANKING_CANDIDATES
        # #         )
        # #
        # #         # Apply re-ranking asynchronously
        # #         if initial_results:
        # #             logger.info(f"Applying re-ranking to {len(initial_results)} context-aware results")
        # #             results = await self.document_manager.rerank_results(question, initial_results, top_k=max_results) # Await async call
        # #             return results
        # #
        # #     # If re-ranking is disabled or failed, use standard search (already got initial_results via custom_search_with_embedding)
        # #     results = self.document_manager.custom_search_with_embedding(embedding, top_k=max_results)
        # #     return results
        # # else:
        # #     # Fallback to normal search with re-ranking asynchronously
        # #     logger.info("No context found, using standard search with re-ranking")
        # #     search_results = await self.document_manager.search( # Await async call
        # #         question,
        # #         top_k=max_results,
        # #         apply_reranking=self.config.RERANKING_ENABLED
        # #     )
        # #     return search_results
        #
        #     # Search with this embedding and apply re-ranking
        #     # apply_reranking = self.config.RERANKING_ENABLED # Part of old logic
        #     #
        #     # if apply_reranking: # Part of old logic
        #     #     # Get more initial results for re-ranking
        #     #     initial_results = self.document_manager.custom_search_with_embedding( # Part of old logic
        #     #         embedding,
        #     #         top_k=self.config.RERANKING_CANDIDATES
        #     #     )
        #     #
        #     #     # Apply re-ranking asynchronously
        #     #     if initial_results: # Part of old logic
        #     #         logger.info(f"Applying re-ranking to {len(initial_results)} context-aware results")
        #     #         results = await self.document_manager.rerank_results(question, initial_results, top_k=max_results) # Await async call
        #     #         return results
        #     #
        #     # # If re-ranking is disabled or failed, use standard search (already got initial_results via custom_search_with_embedding)
        #     # results = self.document_manager.custom_search_with_embedding(embedding, top_k=max_results) # Part of old logic
        #     # return results
        # # else: # Part of old logic (fallback when no history)
        # #     # Fallback to normal search with re-ranking asynchronously
        # #     logger.info("No context found, using standard search with re-ranking") # Part of old logic
        # #     search_results = await self.document_manager.search( # Await async call # Part of old logic
        # #         question,
        # #         top_k=max_results,
        # #         apply_reranking=self.config.RERANKING_ENABLED
        # #     )
        # #     return search_results # Part of old logic

    async def on_message(self, message: discord.Message):
        """Handle incoming messages, ignoring banned users."""
        try:
            # Process commands first
            await self.process_commands(message)

            # Ignore messages from self
            if message.author == self.user:
                return
            
            if "@everyone" in message.content or "@here" in message.content:
                print(f"Ignoring message with @everyone/@here ping from {message.author}: {message.content}")
                return

            # Ignore messages from banned users
            if message.author.id in self.banned_users:
                logger.info(f"Ignored message from banned user {message.author.name} (ID: {message.author.id})")
                return

            # Only respond to mentions
            if not self.user.mentioned_in(message):
                return

            channel_name = message.channel.name if message.guild else "DM"

            # Check if the message is a reply and get the referenced message
            referenced_message = None
            if message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                # Ensure the resolved reference is actually a message object
                referenced_message = message.reference.resolved
                logger.info(f"Message is a reply to a message from {referenced_message.author.name}: {shorten(referenced_message.content, width=100, placeholder='...')}")
            elif message.reference and message.reference.message_id:
                # If resolved is not a message object (might happen if message is old/deleted), try fetching
                try:
                    referenced_message = await message.channel.fetch_message(message.reference.message_id)
                    logger.info(f"Fetched replied-to message from {referenced_message.author.name}: {shorten(referenced_message.content, width=100, placeholder='...')}")
                except discord.NotFound:
                    logger.warning(f"Could not fetch replied-to message (ID: {message.reference.message_id}), might be deleted.")
                except discord.Forbidden:
                    logger.warning(f"No permission to fetch replied-to message (ID: {message.reference.message_id}).")
                except Exception as e:
                    logger.error(f"Error fetching replied-to message (ID: {message.reference.message_id}): {e}")


            logger.info(f"Processing message from {message.author.name} (ID: {message.author.id}): {shorten(message.content, width=100, placeholder='...')}")

            # Extract the question from the message (remove mentions)
            question = message.content
            for mention in message.mentions:
                question = question.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
            question = question.strip()

            # Check if the stripped question is empty
            if not question:
                question = "Hello" # Default to a simple greeting if message is just a ping
                logger.info("Received empty message after stripping mentions, defaulting to 'Hello'")

            # Add context-aware query enhancement (if enabled, based on hybrid search logic)
            original_question = question
            # Context checking and enhancement happens within process_hybrid_query now

            # Check for Google Doc links in the message
            google_doc_ids = await self._extract_google_doc_ids(question)
            google_doc_contents = []

            # --- Image Processing ---
            direct_image_attachments = [] # Images attached directly to the user's message
            referenced_image_attachments = [] # Images attached to the message being replied to
            thinking_msg = await message.channel.send(
                "*neural pathways activating... processing query...*",
                reference=message,
                mention_author=False
            ) # Send initial thinking message

            # Process DIRECT image attachments
            if message.attachments:
                await thinking_msg.edit(content="*neural pathways activating... processing query and analyzing direct images...*")
                for attachment in message.attachments:
                    if is_image(attachment):
                        base64_image = await self._download_image_to_base64(attachment)
                        if base64_image:
                            direct_image_attachments.append(base64_image)
                            logger.info(f"Processed direct image attachment: {attachment.filename}")

            # Process image attachments from the REPLIED-TO message
            if referenced_message and referenced_message.attachments:
                # Update thinking message based on whether direct images were also found
                current_thinking_content = thinking_msg.content # Get current content directly

                if direct_image_attachments:
                    new_thinking_content = "*neural pathways activating... processing query, direct images, and images from reply...*"
                else:
                    new_thinking_content = "*neural pathways activating... processing query and analyzing images from reply...*"
                # Only edit if the content needs changing
                if current_thinking_content != new_thinking_content:
                    try: # Add try-except around edits in case the message was deleted
                        await thinking_msg.edit(content=new_thinking_content)
                    except discord.NotFound:
                        logger.warning("Thinking message was deleted before edit for referenced images.")
                    except Exception as edit_err:
                        logger.error(f"Error editing thinking message for referenced images: {edit_err}")


                for attachment in referenced_message.attachments:
                    if is_image(attachment):
                        base64_image = await self._download_image_to_base64(attachment)
                        if base64_image:
                            referenced_image_attachments.append(base64_image)
                            logger.info(f"Processed referenced image attachment: {attachment.filename}")

            # Combine all image attachments for the API call
            all_api_image_attachments = direct_image_attachments + referenced_image_attachments
            # --- End Image Processing ---

            # Get conversation history for context
            conversation_messages = self.conversation_manager.get_conversation_messages(message.author.name)

            # Get user's preferred model
            preferred_model = self.user_preferences_manager.get_preferred_model(
                str(message.author.id),
                default_model=self.config.LLM_MODEL
            )

            # Update thinking message before search
            await thinking_msg.edit(content="*analyzing query and searching imperial databases...*")

            # Use the new hybrid search system asynchronously
            # This now handles context checking and enhancement internally
            search_results = await self.process_hybrid_query( # Await async call
                question, # Pass the potentially non-enhanced question here
                message.author.name,
                max_results=self.config.get_top_k_for_model(preferred_model),
                use_context=True # Enable context features for on_message
            )

            # Log the results
            logger.info(f"Found {len(search_results)} relevant document sections via hybrid search")

            # Load Google Doc ID mapping for citation links
            googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

            # Extract image IDs from search results
            image_ids = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
                    logger.info(f"Found relevant image from search: {image_id}")

            # Fetch content for Google Doc links found in the *original* question
            if google_doc_ids:
                await thinking_msg.edit(content="*detected Google Doc links in your query... fetching content...*")
                for doc_id, doc_url in google_doc_ids:
                    content = await self._fetch_google_doc_content(doc_id)
                    if content:
                        logger.info(f"Fetched content from linked Google Doc {doc_id}")
                        google_doc_contents.append((doc_id, doc_url, content))

            # Format raw results with citation info
            import urllib.parse
            raw_doc_contexts = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id:
                    image_name = self.image_manager.metadata.get(image_id, {}).get('name', "Unknown Image")
                    raw_doc_contexts.append(f"Image: {image_name} (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}")
                elif doc in googledoc_mapping:
                    doc_id = googledoc_mapping[doc]
                    words = chunk.split()
                    search_text = ' '.join(words[:min(10, len(words))]) # Use first 10 words for search context
                    encoded_search = urllib.parse.quote(search_text)
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/"
                    raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
                else:
                    raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) (similarity: {score:.2f}):\n{chunk}")

            # Add fetched Google Doc content to context
            google_doc_context_str = []
            for doc_id, doc_url, content in google_doc_contents:
                # Truncate content if it's too long (e.g., first 10000 chars)
                truncated_content = content[:10000] + ("..." if len(content) > 10000 else "")
                google_doc_context_str.append(f"From Google Doc URL: {doc_url}:\n{truncated_content}")

            # Get nickname or username
            nickname = message.author.nick if (message.guild and message.author.nick) else message.author.name

            # Determine which system prompt to use based on user preference
            use_informational_prompt = self.user_preferences_manager.get_informational_prompt_mode(str(message.author.id))
            selected_system_prompt = INFORMATIONAL_SYSTEM_PROMPT if use_informational_prompt else SYSTEM_PROMPT
            logger.info(f"Using {'Informational' if use_informational_prompt else 'Standard'} System Prompt for user {message.author.id}")

            # Fetch user pronouns
            pronouns = self.user_preferences_manager.get_pronouns(str(message.author.id))
            pronoun_context_message = None
            if pronouns:
                logger.info(f"User {message.author.id} ({nickname}) has pronouns set: {pronouns}")
                pronoun_context_message = {
                    "role": "system",
                    "content": f"User Information: The user you are interacting with ({nickname}) uses the pronouns '{pronouns}'. Please use these pronouns when referring to the user."
                }
            else:
                 logger.info(f"User {message.author.id} ({nickname}) has no pronouns set.")


            # --- Prepare messages for AI Model ---
            messages = [
                {
                    "role": "system",
                    "content": selected_system_prompt # Use the selected prompt
                }
                # Pronoun context will be inserted here if available
            ]

            # Insert pronoun context if it exists
            if pronoun_context_message:
                messages.insert(1, pronoun_context_message)

            # Add conversation history *after* potential pronoun context
            messages.extend(conversation_messages)


            if referenced_message:
                # Get the author object from the referenced message
                ref_author = referenced_message.author

                # Determine the display name (nickname if available in guild context, otherwise username)
                reply_author_name = ref_author.name # Default to username
                if isinstance(ref_author, discord.Member) and ref_author.nick:
                    # If the author object is a Member AND has a guild nickname set
                    reply_author_name = ref_author.nick

                ref_content = referenced_message.content

                # Sanitize mentions in the referenced message content
                for mention in referenced_message.mentions:
                    # Determine mention display name safely
                    mention_display_name = mention.name # Default to username
                    if isinstance(mention, discord.Member) and mention.nick:
                        mention_display_name = mention.nick
                    ref_content = ref_content.replace(f'<@{mention.id}>', f'@{mention_display_name}').replace(f'<@!{mention.id}>', f'@{mention_display_name}')

                # Add note about attachments in the referenced message, using image_note
                attachment_info = ""
                if referenced_message.attachments:
                    attachment_count = len(referenced_message.attachments)
                    # Define image_note based on processed referenced images
                    image_note = f", including {len(referenced_image_attachments)} image{'s' if len(referenced_image_attachments) > 1 else ''} provided" if referenced_image_attachments else ""
                    # Construct attachment_info correctly using image_note
                    attachment_info = f" [with {attachment_count} attachment{'s' if attachment_count > 1 else ''}{image_note}]"


                # Frame the reply context for the AI
                role_context = "your previous message" if ref_author == self.user else f"a message from {reply_author_name}" # Use the determined name
                messages.append({
                    "role": "system",
                    "content": f"The user is replying to {role_context}: \"{ref_content}\"{attachment_info}"
                })

            # Add raw document context from search results
            if raw_doc_contexts:
                raw_doc_context_combined = "\n\n".join(raw_doc_contexts)
                # Truncate if excessively long to avoid huge prompts
                max_raw_context_len = 32000 # Adjust as needed
                if len(raw_doc_context_combined) > max_raw_context_len:
                    raw_doc_context_combined = raw_doc_context_combined[:max_raw_context_len] + "\n... [Context Truncated]"
                    logger.warning(f"Raw document context truncated to {max_raw_context_len} characters.")
                messages.append({
                    "role": "system",
                    "content": f"Raw document context (with citation links):\n{raw_doc_context_combined}"
                })

            # Add fetched Google Doc content if available
            if google_doc_context_str:
                messages.append({
                    "role": "system",
                    "content": f"Content from Google Docs linked in the query:\n\n{'\n\n'.join(google_doc_context_str)}"
                })

            # Add channel context
            messages.append({
                "role": "system",
                "content": f"You are responding to a message in the Discord channel: {channel_name}"
            })

            # Add image context summary system message
            total_api_images = len(image_ids) + len(all_api_image_attachments)
            if total_api_images > 0:
                img_source_parts = []
                if image_ids: img_source_parts.append(f"{len(image_ids)} from search")
                if direct_image_attachments: img_source_parts.append(f"{len(direct_image_attachments)} attached")
                if referenced_image_attachments: img_source_parts.append(f"{len(referenced_image_attachments)} from reply")
                messages.append({
                    "role": "system",
                    "content": f"The query context includes {total_api_images} image{'s' if total_api_images > 1 else ''} ({', '.join(img_source_parts)}). Vision models will see these in the user message."
                })

            # Finally, add the user's actual message
            messages.append({
                "role": "user",
                "content": f"{nickname}: {original_question}" # Use original question here for clarity
            })
            # --- End Preparing Messages ---

            # Get friendly model name for status updates
            model_name = "Unknown Model"
            if "deepseek/deepseek-r1" in preferred_model: model_name = "DeepSeek-R1"
            elif "deepseek/deepseek-chat" in preferred_model: model_name = "DeepSeek V3 0324"
            elif "google/gemini-2.5-flash" in preferred_model: model_name = "Gemini 2.5 Flash" # Specific check
            elif preferred_model.startswith("google/"): model_name = "Gemini 2.0 Flash" # Fallback for other google models
            elif preferred_model.startswith("nousresearch/"): model_name = "Nous: Hermes 405B"
            elif "claude-3.5-haiku" in preferred_model: model_name = "Claude 3.5 Haiku"
            elif "claude-3.5-sonnet" in preferred_model: model_name = "Claude 3.5 Sonnet"
            elif "claude-3.7-sonnet" in preferred_model: model_name = "Claude 3.7 Sonnet"
            elif "maverick" in preferred_model: model_name = "Llama 4 Maverick"
            elif "qwen/qwq-32b" in preferred_model: model_name = "Qwen QwQ 32B"
            elif "eva-unit-01/eva-qwen-2.5-72b" in preferred_model: model_name = "EVA Qwen 2.5 72B"
            elif "latitudegames/wayfarer" in preferred_model: model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in preferred_model: model_name = "Anubis Pro 105B"
            elif "grok" in preferred_model: model_name = "Grok 3 Mini"
            elif "4.1-mini" in preferred_model:
                model_name = "4.1 Mini"
            elif "4.1-nano" in preferred_model:
                model_name = "4.1 Nano"
            # Note: "Testing Model" name is less clear, using specific names if possible.

            # Update thinking message before API call
            status_update = f"*formulating response with enhanced neural mechanisms using {model_name}...*"
            if total_api_images > 0 and preferred_model not in self.vision_capable_models:
                status_update += f"\n(note: preferred model ({model_name}) doesn't support images. Text only.)"
            elif google_doc_contents:
                status_update += f"\n(using content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})"
            await thinking_msg.edit(content=status_update)

            # Calculate dynamic temperature
            temperature = self.calculate_dynamic_temperature(
                original_question, # Use original question for temperature calculation
                conversation_messages
            )

            # Get AI response
            completion, actual_model = await self._try_ai_completion(
                preferred_model,
                messages,
                image_ids=image_ids, # From search
                image_attachments=all_api_image_attachments, # Combined direct + referenced
                temperature=temperature
            )

            # Process and send response
            if completion and completion.get('choices') and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                    response = completion['choices'][0]['message']['content']
                else:
                    logger.error(f"Unexpected response structure: {completion}")
                    await thinking_msg.edit(content="*neural circuit overload!* I received an unexpected response structure.")
                    return

                # Update conversation history with context notes
                user_history_content = original_question
                history_notes = []
                if direct_image_attachments: history_notes.append("direct images")
                if referenced_message: history_notes.append("reply context")
                if referenced_image_attachments: history_notes.append("reply images")
                if history_notes: user_history_content += f" [{', '.join(history_notes)}]"

                self.conversation_manager.write_conversation(
                    message.author.name, "user", user_history_content, channel_name
                )
                self.conversation_manager.write_conversation(
                    message.author.name, "assistant", response, channel_name
                )

                # Send the response, replacing thinking message
                await self.send_split_message(
                    message.channel,
                    response,
                    reference=message,
                    mention_author=False,
                    model_used=actual_model,
                    user_id=str(message.author.id),
                    existing_message=thinking_msg # Pass the thinking message to edit/replace
                )
            else:
                await thinking_msg.edit(content="*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback
            try:
                # Try editing the thinking message first if it exists
                if 'thinking_msg' in locals() and thinking_msg and isinstance(thinking_msg, discord.Message):
                    await thinking_msg.edit(content="*neural circuit overload!* My brain is struggling and an error has occurred.")
                else: # Fallback to sending a new message
                    await message.channel.send(
                        "*neural circuit overload!* My brain is struggling and an error has occurred.",
                        reference=message,
                        mention_author=False
                    )
            except Exception as send_error:
                logger.error(f"Failed to send error message to user: {send_error}")

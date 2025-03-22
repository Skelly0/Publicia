from __future__ import annotations
import os
import json
import logging
import asyncio
import aiohttp
import discord
import sys
import io
import re
import uuid
import pickle
import time
import random
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from textwrap import shorten
from collections import deque
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from discord import app_commands
from discord.ext import commands
from system_prompt import SYSTEM_PROMPT
from image_prompt import IMAGE_DESCRIPTION_PROMPT
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from rank_bm25 import BM25Okapi



# Reconfigure stdout to use UTF-8 with error replacement
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

async def check_permissions(interaction: discord.Interaction):
    # First check for special user ID (this doesn't require guild permissions)
    if interaction.user.id == 203229662967627777:
        return True
        
    # Check if we're in a guild
    if not interaction.guild:
        raise app_commands.CheckFailure("This command can only be used in a server")
    
    # Try to get permissions directly from interaction.user
    try:
        return interaction.user.guild_permissions.administrator
    except AttributeError:
        # If that fails, try getting the member object
        try:
            member = interaction.guild.get_member(interaction.user.id)
            
            # Member might be None if not in cache
            if member is None:
                # Fetch fresh from API
                member = await interaction.guild.fetch_member(interaction.user.id)
                
            return member.guild_permissions.administrator
        except Exception as e:
            print(f"Permission check error: {e}")
            return False

def is_image(attachment):
    """Check if an attachment is an image based on content type or file extension."""
    if attachment.content_type and attachment.content_type.startswith('image/'):
        return True
    # Fallback to checking file extension
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
    return any(attachment.filename.lower().endswith(ext) for ext in image_extensions)


def split_message(text, max_length=1750):
    """smarter message splitting that respects semantic boundaries"""
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    chunks = []
    current_chunk = ""
    
    # try paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk + ('\n\n' if current_chunk else '') + paragraph) <= max_length:
            current_chunk += ('\n\n' if current_chunk else '') + paragraph
        else:
            # store current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            
            # if paragraph itself is too long
            if len(paragraph) > max_length:
                # try line-by-line
                lines = paragraph.split('\n')
                current_chunk = ""
                
                for line in lines:
                    if len(current_chunk + ('\n' if current_chunk else '') + line) <= max_length:
                        current_chunk += ('\n' if current_chunk else '') + line
                    else:
                        # if line is too long
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        
                        # smart splitting for long lines
                        if len(line) > max_length:
                            # try splitting at these boundaries in order
                            split_markers = ['. ', '? ', '! ', '; ', ', ', ' - ', ' ']
                            
                            start = 0
                            while start < len(line):
                                # find best split point
                                end = start + max_length
                                if end >= len(line):
                                    chunk = line[start:]
                                    if current_chunk and len(current_chunk + chunk) <= max_length:
                                        current_chunk += chunk
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        current_chunk = chunk
                                    break
                                
                                # try each split marker
                                split_point = end
                                for marker in split_markers:
                                    pos = line[start:end].rfind(marker)
                                    if pos > 0:  # found a good split point
                                        split_point = start + pos + len(marker)
                                        break
                                
                                chunk = line[start:split_point]
                                if current_chunk and len(current_chunk + chunk) <= max_length:
                                    current_chunk += chunk
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = chunk
                                
                                start = split_point
                        else:
                            current_chunk = line
            else:
                current_chunk = paragraph
    
    # add final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
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
    """Display super cool ASCII art banner on startup with simple search indicator."""
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
    ║       ** SIMPLE SEARCH MODE ACTIVE - ENHANCED SEARCH DISABLED **  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    # Add color to the banner
    cyan = '\033[36m'
    reset = '\033[0m'
    print(f"{cyan}{banner}{reset}")

    # Display simulation of "neural pathway initialization"
    print(f"{cyan}[INITIATING NEURAL PATHWAYS - SIMPLE SEARCH MODE]{reset}")
    for i in range(10):
        dots = "." * random.randint(3, 10)
        spaces = " " * random.randint(0, 5)
        print(f"{cyan}{spaces}{'>' * (i+1)}{dots} Neural Link {random.randint(1000, 9999)} established{reset}")
        time.sleep(0.2)
    print(f"{cyan}[ALL NEURAL PATHWAYS ACTIVE - ENHANCED SEARCH DISABLED]{reset}")
    print(f"{cyan}[MENTAT INTERFACE READY FOR SERVICE TO THE INFINITE EMPIRE]{reset}\n")
    
    
# Configure logging
logger = configure_logging()

class ImageManager:
    """Manages image storage, descriptions, and retrieval."""
    
    def __init__(self, base_dir: str = "images", document_manager = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference to DocumentManager
        self.document_manager = document_manager
        
        # Storage for image metadata
        self.metadata = {}
        
        # Load existing images
        self._load_images()
    
    def add_image(self, name: str, image_data: bytes, description: str = None):
        """Add a new image to the system with optional description."""
        try:
            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())
            
            # Use .png for all images for simplicity
            image_path = self.base_dir / f"{image_id}.png"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Create metadata
            self.metadata[image_id] = {
                'name': name,
                'added': datetime.now().isoformat(),
                'description': description,
                'path': str(image_path)
            }
            
            # Save metadata
            self._save_metadata()
            
            # Add description as a document for search if provided
            if description and self.document_manager:
                doc_name = f"image_{image_id}.txt"
                self.document_manager.add_document(doc_name, description)
                
                # Add image reference to document metadata
                if doc_name in self.document_manager.metadata:
                    self.document_manager.metadata[doc_name]['image_id'] = image_id
                    self.document_manager.metadata[doc_name]['image_name'] = name
                    self.document_manager._save_to_disk()
            
            return image_id
            
        except Exception as e:
            logger.error(f"Error adding image {name}: {e}")
            raise
    
    def get_image(self, image_id: str) -> Tuple[bytes, str]:
        """Get image data and description by ID."""
        if image_id not in self.metadata:
            raise ValueError(f"Image ID {image_id} not found")
        
        image_path = Path(self.metadata[image_id]['path'])
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        description = self.metadata[image_id].get('description', '')
        
        return image_data, description
    
    def get_base64_image(self, image_id: str) -> str:
        """Get base64-encoded image data by ID."""
        if image_id not in self.metadata:
            raise ValueError(f"Image ID {image_id} not found")
            
        image_path = Path(self.metadata[image_id]['path'])
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} not found")
            
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Default to image/png for simplicity
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
    def _save_metadata(self):
        """Save image metadata to disk."""
        try:
            with open(self.base_dir / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving image metadata: {e}")
            raise
    
    def _load_images(self):
        """Load image metadata from disk."""
        try:
            metadata_path = self.base_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} images")
            else:
                self.metadata = {}
                logger.info("No image metadata found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading image metadata: {e}")
            self.metadata = {}
    
    def list_images(self) -> List[Dict]:
        """List all images with their metadata."""
        return [
            {
                'id': image_id,
                'name': meta['name'],
                'added': meta['added'],
                'has_description': bool(meta.get('description'))
            }
            for image_id, meta in self.metadata.items()
        ]
    
    def delete_image(self, image_id: str) -> bool:
        """Delete an image and its associated description document."""
        if image_id not in self.metadata:
            return False
        
        try:
            # Delete image file
            image_path = Path(self.metadata[image_id]['path'])
            if image_path.exists():
                image_path.unlink()
            
            # Delete description document
            doc_name = f"image_{image_id}.txt"
            if self.document_manager and doc_name in self.document_manager.chunks:
                del self.document_manager.chunks[doc_name]
                del self.document_manager.embeddings[doc_name]
                del self.document_manager.metadata[doc_name]
                self.document_manager._save_to_disk()
            
            # Delete metadata
            del self.metadata[image_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting image {image_id}: {e}")
            return False
            
    def update_description(self, image_id: str, description: str) -> bool:
        """Update the description for an image."""
        if image_id not in self.metadata:
            return False
        
        try:
            # Update metadata
            self.metadata[image_id]['description'] = description
            self._save_metadata()
            
            # Update document
            doc_name = f"image_{image_id}.txt"
            if self.document_manager:
                # Remove old document if it exists
                if doc_name in self.document_manager.chunks:
                    del self.document_manager.chunks[doc_name]
                if doc_name in self.document_manager.embeddings:
                    del self.document_manager.embeddings[doc_name]
                
                # Add new document
                self.document_manager.add_document(doc_name, description)
                
                # Add image reference to document metadata
                if doc_name in self.document_manager.metadata:
                    self.document_manager.metadata[doc_name]['image_id'] = image_id
                    self.document_manager.metadata[doc_name]['image_name'] = self.metadata[image_id]['name']
                    self.document_manager._save_to_disk()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating image description {image_id}: {e}")
            return False

class DocumentManager:
    """Manages document storage, embeddings, and retrieval."""
    
    def __init__(self, base_dir: str = "documents", top_k: int = 5, config=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Store top_k as instance variable
        self.top_k = top_k
        self.config = config
        
        # Initialize Google Generative AI embedding model
        try:
            logger.info("Initializing Google Generative AI embedding model")
            # Check if API key is available
            if not config or not config.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY environment variable not set")
                raise ValueError("GOOGLE_API_KEY environment variable not set")
                
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            
            # Set embedding model
            self.embedding_model = config.EMBEDDING_MODEL if config else 'models/text-embedding-004'
            self.embedding_dimensions = config.EMBEDDING_DIMENSIONS if config and config.EMBEDDING_DIMENSIONS > 0 else None
            
            logger.info(f"Using Google embedding model: {self.embedding_model}")
            if self.embedding_dimensions:
                logger.info(f"Truncating embeddings to {self.embedding_dimensions} dimensions")
            else:
                logger.info("Using full 3072 dimensions. Consider setting EMBEDDING_DIMENSIONS to 1024 or 512 for better storage efficiency.")
            
            logger.info(f"Gemini embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Generative AI: {e}")
            raise
        
        # Storage for documents and embeddings
        self.chunks: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Load existing documents
        logger.info("Starting document loading")
        self._load_documents()
        logger.info("Document loading completed")

        self.bm25_indexes = {}

    def _create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from document chunks."""
        # Tokenize chunks - simple whitespace tokenization
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
        
    def _search_bm25(self, query: str, top_k: int = None) -> List[Tuple[str, str, float, Optional[str], int, int]]:
        """Search for chunks using BM25."""
        if top_k is None:
            top_k = self.top_k
            
        # Tokenize query
        query_tokens = query.lower().split()
        
        results = []
        for doc_name, chunks in self.chunks.items():
            # Make sure we have a BM25 index for this document
            if doc_name not in self.bm25_indexes:
                self.bm25_indexes[doc_name] = self._create_bm25_index(
                    self.contextualized_chunks.get(doc_name, chunks)
                )
                
            # Get BM25 scores
            bm25_scores = self.bm25_indexes[doc_name].get_scores(query_tokens)
            
            # Add top results
            for idx, score in enumerate(bm25_scores):
                if idx < len(chunks):  # Safety check
                    image_id = None
                    if doc_name.startswith("image_") and doc_name.endswith(".txt"):
                        image_id = doc_name[6:-4]
                    elif doc_name in self.metadata and 'image_id' in self.metadata[doc_name]:
                        image_id = self.metadata[doc_name]['image_id']
                    
                    # Use contextualized chunk instead of original
                    chunk = self.get_contextualized_chunk(doc_name, idx)
                    
                    results.append((
                        doc_name,
                        chunk,
                        float(score),
                        image_id,
                        idx + 1,
                        len(chunks)
                    ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def get_contextualized_chunk(self, doc_name: str, chunk_idx: int) -> str:
        """Get the contextualized chunk if available, otherwise fall back to the original chunk.
        
        Args:
            doc_name: The document name
            chunk_idx: The 0-based index of the chunk
            
        Returns:
            The contextualized chunk if available, otherwise the original chunk
        """
        # Check if we have contextualized chunks for this document
        if doc_name in self.contextualized_chunks and chunk_idx < len(self.contextualized_chunks[doc_name]):
            return self.contextualized_chunks[doc_name][chunk_idx]
        
        # Fall back to original chunks
        if doc_name in self.chunks and chunk_idx < len(self.chunks[doc_name]):
            logger.warning(f"Using original chunk for {doc_name}[{chunk_idx}] as contextualized version not found")
            return self.chunks[doc_name][chunk_idx]
        
        # Emergency fallback
        logger.error(f"Chunk not found: {doc_name}[{chunk_idx}]")
        return "Chunk not found"
        
    def _combine_search_results(self, embedding_results: List[Tuple], bm25_results: List[Tuple], top_k: int = None) -> List[Tuple]:
        """Combine results from embedding and BM25 search using score-based fusion."""
        if top_k is None:
            top_k = self.top_k
            
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Log the number of results from each method
        logger.info(f"Combining {len(embedding_results)} embedding results with {len(bm25_results)} BM25 results using score-based fusion")
        
        # Normalize embedding scores if needed - they should already be between 0 and 1 (cosine similarity)
        # But we'll ensure they're properly normalized anyway
        embedding_weight = 0.85  # Weight for embedding scores (adjust as needed)
        
        # Process embedding results 
        for doc_name, chunk, score, image_id, chunk_idx, total_chunks in embedding_results:
            key = (doc_name, chunk_idx)
            # Ensure score is between 0 and 1
            norm_score = max(0, min(1, score))
            combined_scores[key] = combined_scores.get(key, 0) + (norm_score * embedding_weight)
            
            # Log some of the top embedding scores
            if len(combined_scores) <= 3:
                logger.info(f"Top embedding result: {doc_name}, score: {score:.4f}, normalized: {norm_score:.4f}")
        
        # Normalize BM25 scores to [0, 1] range
        bm25_weight = 0.15  # Weight for BM25 scores (adjust as needed)
        
        if bm25_results:
            # Find min and max scores for normalization
            max_bm25 = max(score for _, _, score, _, _, _ in bm25_results)
            min_bm25 = min(score for _, _, score, _, _, _ in bm25_results)
            range_bm25 = max_bm25 - min_bm25
            
            # Log min/max for debugging
            logger.info(f"BM25 score range: min={min_bm25:.4f}, max={max_bm25:.4f}")
        else:
            # Default values if no results
            max_bm25, min_bm25, range_bm25 = 1.0, 0.0, 1.0
        
        # Process BM25 results
        for doc_name, chunk, score, image_id, chunk_idx, total_chunks in bm25_results:
            key = (doc_name, chunk_idx)
            # Normalize BM25 score to [0, 1] range
            if range_bm25 > 0:
                norm_score = (score - min_bm25) / range_bm25
            else:
                norm_score = 0.5  # Default if all scores are the same
                
            combined_scores[key] = combined_scores.get(key, 0) + (norm_score * bm25_weight)
            
            # Log some of the top BM25 scores
            if len([k for k in combined_scores.keys() if k == key]) <= 3:
                logger.info(f"Top BM25 result: {doc_name}, score: {score:.4f}, normalized: {norm_score:.4f}")
        
        # Safety check - ensure we have some scores
        if not combined_scores:
            logger.warning("No combined scores found. Returning empty results.")
            return []
        
        # Create combined results
        combined_results = []
        for (doc_name, chunk_idx), score in combined_scores.items():
            # Use contextualized chunk instead of original
            chunk_index = chunk_idx - 1  # Convert from 1-based to 0-based indexing
            
            # Safety check for valid index
            if chunk_index < 0 or chunk_index >= len(self.chunks.get(doc_name, [])):
                continue
                
            # Get the contextualized chunk
            chunk = self.get_contextualized_chunk(doc_name, chunk_index)
                
            image_id = None
            if doc_name.startswith("image_") and doc_name.endswith(".txt"):
                image_id = doc_name[6:-4]
            elif doc_name in self.metadata and 'image_id' in self.metadata[doc_name]:
                image_id = self.metadata[doc_name]['image_id']
                
            combined_results.append((
                doc_name,
                chunk,
                score,  # This is the combined score
                image_id,
                chunk_idx,
                len(self.chunks[doc_name])
            ))
        
        # Sort by score and return top_k
        combined_results.sort(key=lambda x: x[2], reverse=True)
        
        # Log top combined results
        for i, (doc_name, _, score, _, _, _) in enumerate(combined_results[:3]):
            logger.info(f"Top {i+1} combined result: {doc_name}, score: {score:.4f}")
        
        return combined_results[:top_k]

    async def generate_chunk_context(self, document_content: str, chunk_content: str) -> str:
        """Generate context for a chunk using Gemini 2.0 Flash via OpenRouter."""
        try:
            # Create the prompt as messages for the API
            system_prompt = """You are a helpful AI assistant that creates concise contextual descriptions for document chunks. Your task is to provide a short, succinct context that situates a specific chunk within the overall document to improve search retrieval. Answer only with the succinct context and nothing else."""
            
            user_prompt = f"""
            <document> 
            {document_content} 
            </document> 
            Here is the chunk we want to situate within the whole document 
            <chunk> 
            {chunk_content} 
            </chunk> 
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # OpenRouter API headers - same as used elsewhere in the bot
            headers = {
                "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://discord.com", 
                "X-Title": "Publicia - Context Generation",
                "Content-Type": "application/json"
            }
            
            # Prepare payload
            payload = {
                "model": "google/gemini-2.0-flash-001",
                "messages": messages,
                "temperature": 0.1,  # Low temperature for deterministic outputs
                "max_tokens": 150    # Limit token length
            }
            
            # Make API call
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=360  # 60 second timeout
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(f"API error (Status {response.status}): {error_text}")
                                # If we had a non-200 status, try again or return default
                                if attempt == max_retries - 1:
                                    return f"This chunk is from document dealing with {document_content[:50]}..."
                                continue
                                
                            completion = await response.json()
                    
                    # Extract context from the response
                    if completion and completion.get('choices') and len(completion['choices']) > 0:
                        if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                            context = completion['choices'][0]['message']['content'].strip()
                            
                            # Ensure context isn't too long (target ~50-100 tokens)
                            """
                            if len(context.split()) > 150:
                                # If too long, truncate and add ellipsis
                                words = context.split()
                                context = " ".join(words[:150]) + "..."
                            """
                            
                            logger.info(f"Generated context ({len(context.split())} words): {context[:50]}...")
                            return context
                    
                    # If we couldn't extract a valid context, return default
                    logger.error("Failed to extract valid context")
                    return f"This chunk is from document dealing with {document_content[:50]}..."
                    
                except Exception as e:
                    logger.error(f"Error generating chunk context (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        # If we've used all retries, return default context
                        return f"This chunk is from document dealing with {document_content[:50]}..."
                    # Otherwise, continue to next retry
                    await asyncio.sleep(1)  # Wait before retrying
            
            # This should never be reached due to the returns in the loop, but just in case
            return f"This chunk is from document dealing with {document_content[:50]}..."
            
        except Exception as e:
            logger.error(f"Unhandled error generating chunk context: {e}")
            # Return a default context if generation fails
            return f"This chunk is from document dealing with {document_content[:50]}..."
    
    async def contextualize_chunks(self, doc_name: str, document_content: str, chunks: List[str]) -> List[str]:
        """Generate context for each chunk and prepend to the chunk."""
        contextualized_chunks = []
        
        logger.info(f"Contextualizing {len(chunks)} chunks for document: {doc_name}")
        for i, chunk in enumerate(chunks):
            # Generate context
            context = await self.generate_chunk_context(document_content, chunk)
            
            # Prepend context to chunk
            contextualized_chunk = f"{context} {chunk}"
            
            contextualized_chunks.append(contextualized_chunk)
            
            # Log progress for longer documents
            if (i + 1) % 10 == 0:
                logger.info(f"Contextualized {i + 1}/{len(chunks)} chunks for {doc_name}")
        
        return contextualized_chunks

    def generate_embeddings(self, texts: List[str], is_query: bool = False, titles: List[str] = None) -> np.ndarray:
        """Generate embeddings for a list of text chunks using Google's Generative AI."""
        try:
            import google.generativeai as genai
            
            embeddings = []
            task_type = "retrieval_query" if is_query else "retrieval_document"
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    # If we've generated at least one embedding, use zeros with same dimension
                    if embeddings:
                        zero_embedding = np.zeros_like(embeddings[0])
                        embeddings.append(zero_embedding)
                        continue
                    else:
                        # Otherwise, skip this empty text
                        logger.warning("Skipping empty text for embedding")
                        continue
                
                # Prepare embedding request
                params = {
                    "model": self.embedding_model,
                    "content": text,
                    "task_type": task_type
                }
                
                # Add title if available and it's a document (not a query)
                if not is_query and titles and i < len(titles):
                    params["title"] = titles[i]
                
                result = genai.embed_content(**params)
                
                # Extract the embedding from the result
                embedding_vector = np.array(result["embedding"])
                
                # Truncate if dimensions is specified (MRL allows efficient truncation)
                if self.embedding_dimensions and self.embedding_dimensions < len(embedding_vector):
                    embedding_vector = embedding_vector[:self.embedding_dimensions]
                
                embeddings.append(embedding_vector)
                
            return np.array(embeddings)
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        # Use config values if available, otherwise use defaults
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE if self.config else 750
        if overlap is None:
            overlap = self.config.CHUNK_OVERLAP if self.config else 125
        
        # Handle empty text
        if not text or not text.strip():
            return []
            
        words = text.split()
        if not words:
            return []
        
        if len(words) <= chunk_size:
            return [' '.join(words)]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            end_idx = min(i + chunk_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
            
        logger.info(f"Chunked text into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def cleanup_empty_documents(self):
        """Remove any documents with empty embeddings from the system."""
        empty_docs = []
        for doc_name, doc_embeddings in self.embeddings.items():
            if len(doc_embeddings) == 0:
                empty_docs.append(doc_name)
        
        for doc_name in empty_docs:
            logger.info(f"Removing empty document: {doc_name}")
            del self.chunks[doc_name]
            del self.embeddings[doc_name]
            if doc_name in self.metadata:
                del self.metadata[doc_name]
        
        if empty_docs:
            logger.info(f"Removed {len(empty_docs)} empty documents")
            self._save_to_disk()
        
        return empty_docs
    
    async def add_document(self, name: str, content: str, save_to_disk: bool = True):
        """Add a new document to the system with contextual retrieval."""
        try:
            if not content or not content.strip():
                logger.warning(f"Document {name} has no content. Skipping.")
                return
                
            # Create chunks
            chunks = self._chunk_text(content)
            if not chunks:
                logger.warning(f"Document {name} has no content to chunk. Skipping.")
                return
            
            # Generate contextualized chunks
            contextualized_chunks = await self.contextualize_chunks(name, content, chunks)
            
            # Generate embeddings using contextualized chunks
            # Use document name as title for all chunks to improve embedding quality
            titles = [name] * len(contextualized_chunks)
            embeddings = self.generate_embeddings(contextualized_chunks, is_query=False, titles=titles)
            
            # Store document data
            self.chunks[name] = chunks  # Store original chunks
            self.contextualized_chunks[name] = contextualized_chunks  # Store contextualized chunks
            self.embeddings[name] = embeddings
            self.metadata[name] = {
                'added': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }
            
            # Create BM25 index for contextualized chunks
            self.bm25_indexes[name] = self._create_bm25_index(contextualized_chunks)
            
            # Save to disk only if requested
            if save_to_disk:
                self._save_to_disk()
            
            logger.info(f"Added document: {name} with {len(chunks)} chunks and contextualized embeddings")
            
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
    
    def search(self, query: str, top_k: int = None, apply_reranking: bool = None) -> List[Tuple[str, str, float, Optional[str], int, int]]:
        """
        Search for relevant document chunks with optional re-ranking.
        
        Args:
            query: The search query
            top_k: Number of final results to return
            apply_reranking: Whether to apply re-ranking (overrides config setting)
            
        Returns:
            List of tuples (doc_name, chunk, similarity_score, image_id_if_applicable, chunk_index, total_chunks)
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            # Determine whether to apply re-ranking
            if apply_reranking is None and self.config:
                apply_reranking = self.config.RERANKING_ENABLED
            
            if not query or not query.strip():
                logger.warning("Empty query provided to search")
                return []
                
            # Generate query embedding
            query_embedding = self.generate_embeddings([query], is_query=True)[0]
            
            # Determine how many initial results to retrieve
            initial_top_k = self.config.RERANKING_CANDIDATES if apply_reranking and self.config else top_k
            initial_top_k = max(initial_top_k, top_k)  # Always get at least top_k results
            
            # Get initial results from embeddings
            embedding_results = self.custom_search_with_embedding(query_embedding, top_k=initial_top_k)
            
            # Get results from BM25
            logger.info("Performing BM25 search")
            bm25_results = self._search_bm25(query, top_k=initial_top_k)
            
            # Combine results
            logger.info(f"Combining {len(embedding_results)} embedding results with {len(bm25_results)} BM25 results")
            combined_results = self._combine_search_results(embedding_results, bm25_results, top_k=initial_top_k)
            
            # Apply re-ranking if enabled and we have enough results
            if apply_reranking and len(combined_results) > 1:  
                logger.info(f"Applying re-ranking to {len(combined_results)} initial results")
                return self.rerank_results(query, combined_results, top_k=top_k)
            else:
                # No re-ranking, return initial results
                logger.info(f"Skipping re-ranking (enabled={apply_reranking}, results={len(combined_results)})")
                return combined_results[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def custom_search_with_embedding(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[str, str, float, Optional[str], int, int]]:
        """Search using a pre-generated embedding instead of creating one from text."""
        if top_k is None:
            top_k = self.top_k
        
        results = []
        logger.info("Performing custom search with pre-generated embedding")
        
        for doc_name, doc_embeddings in self.embeddings.items():
            if len(doc_embeddings) == 0:
                logger.warning(f"Skipping document {doc_name} with empty embeddings")
                continue
                
            # Calculate similarities (dot product since embeddings are normalized)
            similarities = np.dot(doc_embeddings, query_embedding)
            
            if len(similarities) > 0:
                top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):]
                
                for idx in top_indices:
                    image_id = None
                    if doc_name.startswith("image_") and doc_name.endswith(".txt"):
                        image_id = doc_name[6:-4]
                    elif doc_name in self.metadata and 'image_id' in self.metadata[doc_name]:
                        image_id = self.metadata[doc_name]['image_id']
                    
                    if idx < len(self.chunks[doc_name]):
                        # Use contextualized chunk instead of original
                        chunk = self.get_contextualized_chunk(doc_name, idx)
                        
                        results.append((
                            doc_name,
                            chunk,
                            float(similarities[idx]),
                            image_id,
                            idx + 1,  # 1-based indexing for display
                            len(self.chunks[doc_name])
                        ))
        
        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Log search results
        for doc_name, chunk, similarity, image_id, chunk_index, total_chunks in results[:top_k]:
            logger.info(f"Found relevant chunk in {doc_name} (similarity: {similarity:.2f}, chunk: {chunk_index}/{total_chunks})")
            if image_id:
                logger.info(f"This is an image description for image ID: {image_id}")
            
            # Log whether we're using a contextualized chunk
            is_contextualized = (doc_name in self.contextualized_chunks and 
                                chunk_index - 1 < len(self.contextualized_chunks[doc_name]) and
                                chunk == self.contextualized_chunks[doc_name][chunk_index - 1])
            logger.info(f"Using {'contextualized' if is_contextualized else 'original'} chunk")
            
            logger.info(f"Chunk content: {shorten(sanitize_for_logging(chunk), width=300, placeholder='...')}")
        
        return results[:top_k]
    
    def _save_to_disk(self):
        """Save document data to disk."""
        try:
            # Save chunks
            with open(self.base_dir / 'chunks.pkl', 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save contextualized chunks
            with open(self.base_dir / 'contextualized_chunks.pkl', 'wb') as f:
                pickle.dump(self.contextualized_chunks, f)
            
            # Save embeddings
            with open(self.base_dir / 'embeddings.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with open(self.base_dir / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f)
                
            # Save embedding provider info
            with open(self.base_dir / 'embeddings_provider.txt', 'w') as f:
                f.write("gemini")
                
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
            raise

    def regenerate_all_embeddings(self):
        """Regenerate all embeddings using the current embedding model."""
        try:
            logger.info("Starting regeneration of all embeddings")
            
            # Iterate through all documents
            for doc_name, chunks in self.chunks.items():
                logger.info(f"Regenerating embeddings for document: {doc_name}")
                
                # Generate new embeddings
                titles = [doc_name] * len(chunks)
                new_embeddings = self.generate_embeddings(chunks, titles=titles)
                
                # Update stored embeddings
                self.embeddings[doc_name] = new_embeddings
            
            # Save to disk
            self._save_to_disk()
            
            logger.info("Completed regeneration of all embeddings")
            return True
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}")
            return False
    
    async def _load_documents(self, force_reload: bool = False):
        """Load document data from disk and add any new .txt files."""
        try:
            # Check if embeddings provider has changed
            embeddings_provider_file = self.base_dir / 'embeddings_provider.txt'
            provider_changed = False
            
            if embeddings_provider_file.exists():
                with open(embeddings_provider_file, 'r') as f:
                    stored_provider = f.read().strip()
                    if stored_provider != "gemini":
                        logger.warning(f"Embedding provider changed from {stored_provider} to gemini")
                        logger.warning("All embeddings will be regenerated to ensure compatibility")
                        provider_changed = True
            
            if not force_reload and not provider_changed and (self.base_dir / 'chunks.pkl').exists():
                # Load existing processed data
                with open(self.base_dir / 'chunks.pkl', 'rb') as f:
                    self.chunks = pickle.load(f)
                
                # Load contextualized chunks if available
                if (self.base_dir / 'contextualized_chunks.pkl').exists():
                    with open(self.base_dir / 'contextualized_chunks.pkl', 'rb') as f:
                        self.contextualized_chunks = pickle.load(f)
                else:
                    # If no contextualized chunks, initialize empty
                    self.contextualized_chunks = {}
                    logger.info("No contextualized chunks found, will generate when needed")
                
                if provider_changed:
                    # If provider changed, only load chunks and regenerate embeddings
                    logger.info("Provider changed, regenerating embeddings for all documents")
                    self.embeddings = {}
                    self.metadata = {}
                    self.bm25_indexes = {}
                    
                    # Regenerate embeddings for all documents
                    for doc_name, chunks in self.chunks.items():
                        logger.info(f"Regenerating embeddings for document: {doc_name}")
                        
                        # Get original document content if available
                        doc_path = self.base_dir / doc_name
                        if doc_path.exists():
                            with open(doc_path, 'r', encoding='utf-8-sig') as f:
                                doc_content = f.read()
                        else:
                            # If original not available, concatenate chunks
                            doc_content = " ".join(chunks)
                        
                        # Generate contextualized chunks
                        contextualized_chunks = await self.contextualize_chunks(doc_name, doc_content, chunks)
                        self.contextualized_chunks[doc_name] = contextualized_chunks
                        
                        # Generate embeddings
                        titles = [doc_name] * len(contextualized_chunks)
                        self.embeddings[doc_name] = self.generate_embeddings(contextualized_chunks, is_query=False, titles=titles)
                        
                        # Create BM25 index
                        self.bm25_indexes[doc_name] = self._create_bm25_index(contextualized_chunks)
                        
                        self.metadata[doc_name] = {
                            'added': datetime.now().isoformat(),
                            'chunk_count': len(chunks)
                        }
                else:
                    # Normal load if provider hasn't changed
                    with open(self.base_dir / 'embeddings.pkl', 'rb') as f:
                        self.embeddings = pickle.load(f)
                    with open(self.base_dir / 'metadata.json', 'r') as f:
                        self.metadata = json.load(f)
                    
                    # Create BM25 indexes for all documents
                    self.bm25_indexes = {}
                    for doc_name, chunks in self.chunks.items():
                        # Use contextualized chunks if available, otherwise use original chunks
                        if doc_name in self.contextualized_chunks:
                            self.bm25_indexes[doc_name] = self._create_bm25_index(self.contextualized_chunks[doc_name])
                        else:
                            # Generate contextualized chunks
                            doc_path = self.base_dir / doc_name
                            if doc_path.exists():
                                with open(doc_path, 'r', encoding='utf-8-sig') as f:
                                    doc_content = f.read()
                            else:
                                # If original not available, concatenate chunks
                                doc_content = " ".join(chunks)
                            
                            contextualized_chunks = await self.contextualize_chunks(doc_name, doc_content, chunks)
                            self.contextualized_chunks[doc_name] = contextualized_chunks
                            self.bm25_indexes[doc_name] = self._create_bm25_index(contextualized_chunks)
                    
                    logger.info(f"Loaded {len(self.chunks)} documents from processed data")
            else:
                # Start fresh if force_reload is True or no processed data exists
                self.chunks = {}
                self.contextualized_chunks = {}
                self.embeddings = {}
                self.metadata = {}
                self.bm25_indexes = {}
                logger.info("Starting fresh or force reloading documents")

            # Save current provider info
            with open(embeddings_provider_file, 'w') as f:
                f.write("gemini")

            # Find .txt files not already loaded
            existing_names = set(self.chunks.keys())
            txt_files = [f for f in self.base_dir.glob('*.txt') if f.name not in existing_names]

            if txt_files:
                logger.info(f"Found {len(txt_files)} new .txt files to load")
                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                        await self.add_document(txt_file.name, content, save_to_disk=False)
                        logger.info(f"Loaded and processed {txt_file.name}")
                    except Exception as e:
                        logger.error(f"Error processing {txt_file.name}: {e}")
            else:
                logger.info("No new .txt files to load")

            # Save the updated state to disk
            self._save_to_disk()
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self.chunks = {}
            self.contextualized_chunks = {}
            self.embeddings = {}
            self.metadata = {}
            self.bm25_indexes = {}

    async def reload_documents(self):
        """Reload all documents from disk, regenerating embeddings."""
        await self._load_documents(force_reload=True)
            
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

    def delete_document(self, name: str) -> bool:
        """Delete a document from the system."""
        try:
            # Check if it's a regular document
            if name in self.chunks:
                # Remove from memory
                del self.chunks[name]
                del self.embeddings[name]
                del self.metadata[name]
                
                # Save changes
                self._save_to_disk()
                
                # Remove file if it exists
                file_path = self.base_dir / name
                if file_path.exists():
                    file_path.unlink()
                    
                return True
                
            # Check if it's a lorebook
            lorebooks_path = self.get_lorebooks_path()
            lorebook_path = lorebooks_path / name
            
            # Try with .txt extension if not found
            if not lorebook_path.exists() and not name.endswith('.txt'):
                lorebook_path = lorebooks_path / f"{name}.txt"
                
            if lorebook_path.exists():
                lorebook_path.unlink()
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error deleting document {name}: {e}")
            return False

    def rerank_results(self, query: str, initial_results: List[Tuple], top_k: int = None) -> List[Tuple]:
        """
        Re-rank search results using the Gemini embedding model for more nuanced relevance.
        
        Args:
            query: The search query
            initial_results: List of tuples (doc_name, chunk, similarity, image_id, chunk_index, total_chunks)
            top_k: Number of results to return after re-ranking
            
        Returns:
            List of re-ranked results
        """
        if top_k is None:
            top_k = self.top_k
            
        if not initial_results:
            return []
        
        logger.info(f"Re-ranking {len(initial_results)} initial results for query: {query}")
        
        # Extract the text chunks from the initial results
        chunks = [result[1] for result in initial_results]
        
        # Create specialized embedding queries for re-ranking
        # This approach creates embeddings that better capture relevance to the specific query
        
        # 1. Generate a query-focused embedding that represents what we're looking for
        query_context = f"Question: {query}\nWhat information would fully answer this question?"
        query_embedding = self.generate_embeddings([query_context], is_query=True)[0]
        
        # 2. Generate content-focused embeddings for each chunk
        content_texts = [f"This document contains the following information: {chunk}" for chunk in chunks]
        content_embeddings = self.generate_embeddings(content_texts, is_query=False)
        
        # 3. Calculate relevance scores using these specialized embeddings
        relevance_scores = np.dot(content_embeddings, query_embedding)
        
        # 4. Create re-ranked results by combining original and new scores
        reranked_results = []
        for i, (doc, chunk, orig_sim, image_id, chunk_idx, total_chunks) in enumerate(initial_results):
            if i < len(relevance_scores):
                # Use weighted combination (favor the re-ranking score)
                combined_score = 0.3 * orig_sim + 0.7 * float(relevance_scores[i])
                reranked_results.append((doc, chunk, combined_score, image_id, chunk_idx, total_chunks))
        
        # 5. Sort by combined score
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        
        # 6. Filter by minimum score threshold
        min_score = self.config.RERANKING_MIN_SCORE if self.config else 0.45
        filtered_results = [r for r in reranked_results if r[2] >= min_score]
        
        # Use filtered results if we have enough, otherwise use all re-ranked results
        filter_mode = self.config.RERANKING_FILTER_MODE if self.config else 'strict'

        # Apply filtering based on mode
        if filter_mode == 'dynamic':
            # Analyze score distribution and set threshold dynamically
            scores = [score for _, _, score, _, _, _ in reranked_results]
            if scores:
                mean = sum(scores) / len(scores)
                # Set threshold to mean * factor (can be tuned)
                dynamic_threshold = mean * 0.8  # 80% of mean
                min_score = max(min_score, dynamic_threshold)
                logger.info(f"Dynamic threshold set to {min_score:.3f} (80% of mean {mean:.3f})")
                filtered_results = [r for r in reranked_results if r[2] >= min_score]
            else:
                filtered_results = []
        elif filter_mode == 'topk':
            # Traditional behavior - get top k regardless of score
            filtered_results = reranked_results[:top_k] if top_k else reranked_results
        else:
            # 'strict' mode - use absolute threshold (already calculated above)
            filtered_results = [r for r in reranked_results if r[2] >= min_score]
        
        # Add fallback mechanism if filtered results are empty
        if not filtered_results:
            logger.warning(f"Filtering with mode '{filter_mode}' removed all results. Falling back to top-{top_k} results.")
            filtered_results = reranked_results[:top_k] if top_k else []
        
        # Only limit to top_k if we have more than needed and filter_mode isn't 'strict'
        if top_k and len(filtered_results) > top_k:
            final_results = filtered_results[:top_k]
            logger.info(f"Enforcing maximum of {top_k} results (had {len(filtered_results)} after filtering)")
        else:
            final_results = filtered_results

        logger.info(f"Re-ranking with '{filter_mode}' mode: {len(reranked_results)} -> {len(final_results)} results")
        
        # Log before/after for comparison
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Re-ranking changed results from {len(initial_results)} to {len(final_results[:top_k])}")
            
            # Log the top 3 results before and after for comparison
            logger.info("Top 3 BEFORE re-ranking:")
            for i, (doc, chunk, sim, img_id, idx, total) in enumerate(initial_results[:3]):
                logger.info(f"  #{i+1}: {doc} (score: {sim:.3f})")
                
            logger.info("Top 3 AFTER re-ranking:")
            for i, (doc, chunk, sim, img_id, idx, total) in enumerate(final_results[:3]):
                logger.info(f"  #{i+1}: {doc} (score: {sim:.3f})")
        
        # Return top_k results
        return final_results[:top_k]

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # Configure models with defaults
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'google/gemini-2.0-flash-001')
        self.CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'google/gemini-2.0-flash-001')
        
        # New embedding configuration
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-004')
        self.EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '0'))
        
        # Chunk size configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '300'))  # Default to 750 words per chunk
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '30'))  # Default to 125 words overlap
        
        # TOP_K configuration with multiplier
        self.TOP_K = int(os.getenv('TOP_K', '5'))
        self.MAX_TOP_K = int(os.getenv('MAX_TOP_K', '20'))
        self.TOP_K_MULTIPLIER = float(os.getenv('TOP_K_MULTIPLIER', '1'))  # Default to no change

        self.MODEL_TOP_K = {
            # DeepSeek models 
            "deepseek/deepseek-r1:free": 20,
            "deepseek/deepseek-r1": 10,
            "deepseek/deepseek-r1-distill-llama-70b": 14,
            "deepseek/deepseek-r1:floor": 10,
            "deepseek/deepseek-r1:nitro": 7,
            "deepseek/deepseek-chat": 10,
            # Gemini models 
            "google/gemini-2.0-flash-001": 17,
            "google/gemini-2.0-pro-exp-02-05:free": 20,
            # Nous Hermes models
            "nousresearch/hermes-3-llama-3.1-405b": 9,
            # Claude models
            "anthropic/claude-3.5-haiku:beta": 9,
            "anthropic/claude-3.5-haiku": 9,
            "anthropic/claude-3.5-sonnet:beta": 5,
            "anthropic/claude-3.5-sonnet": 5,
            "anthropic/claude-3.7-sonnet:beta": 5,
            "anthropic/claude-3.7-sonnet": 5,
            # Qwen models
            "qwen/qwq-32b:free": 20,
            "qwen/qwq-32b": 13, 
            # Testing models
            "thedrummer/unslopnemo-12b": 12,
            "eva-unit-01/eva-qwen-2.5-72b": 9,
            # New models
            "latitudegames/wayfarer-large-70b-llama-3.3": 9,  # good balance for storytelling
            "thedrummer/anubis-pro-105b-v1": 8,  # slightly smaller for this massive model
            # Gemini embedding model
            "models/text-embedding-004": 20,  # Optimized for larger chunks
        }
        
        # Validate required environment variables
        self._validate_config()
        
        # Add timeout settings
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', '180'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '10'))

        self.MODEL_PROVIDERS = {
            "deepseek/deepseek-r1": {
                "order": ["Minimax", "Nebius", "DeepInfra"]
            },
            "eva-unit-01/eva-qwen-2.5-72b": {
                "order": ["Parasail"]
            },
            # Add any other model variants that need custom provider ordering
        }

        self.TEMPERATURE_BASE = float(os.getenv('TEMPERATURE_BASE', '0.1'))
        self.TEMPERATURE_MIN = float(os.getenv('TEMPERATURE_MIN', '0.0'))
        self.TEMPERATURE_MAX = float(os.getenv('TEMPERATURE_MAX', '0.4'))

        self.RERANKING_ENABLED = bool(os.getenv('RERANKING_ENABLED', 'False').lower() in ('true', '1', 'yes'))
        self.RERANKING_CANDIDATES = int(os.getenv('RERANKING_CANDIDATES', '20'))  # Number of initial candidates
        self.RERANKING_MIN_SCORE = float(os.getenv('RERANKING_MIN_SCORE', '0.5'))  # Minimum score threshold

        self.RERANKING_FILTER_MODE = os.getenv('RERANKING_FILTER_MODE', 'strict')  # 'strict', 'dynamic', or 'topk'

        # Validate temperature settings
        if not (0 <= self.TEMPERATURE_MIN <= self.TEMPERATURE_BASE <= self.TEMPERATURE_MAX <= 1):
            logger.warning(f"Invalid temperature settings: MIN({self.TEMPERATURE_MIN}), BASE({self.TEMPERATURE_BASE}), MAX({self.TEMPERATURE_MAX})")
            logger.warning("Temperatures should satisfy: 0 ≤ MIN ≤ BASE ≤ MAX ≤ 1")
            # Fall back to defaults if settings are invalid
            self.TEMPERATURE_BASE = 0.1 
            self.TEMPERATURE_MIN = 0.0
            self.TEMPERATURE_MAX = 0.4
            logger.warning(f"Using fallback temperature settings: MIN({self.TEMPERATURE_MIN}), BASE({self.TEMPERATURE_BASE}), MAX({self.TEMPERATURE_MAX})")
   
    def get_provider_config(self, model: str):
        # Extract base model without suffixes like :free or :nitro
        base_model = model.split(':')[0] if ':' in model else model
        
        # First try exact match, then try base model
        return self.MODEL_PROVIDERS.get(model) or self.MODEL_PROVIDERS.get(base_model)
        
    def get_top_k_for_model(self, model: str) -> int:
        """Get the top_k value for a specific model, applying the configured multiplier."""
        # First try to match the exact model name (including any suffixes)
        if model in self.MODEL_TOP_K:
            base_top_k = self.MODEL_TOP_K[model]
        else:
            # If not found, extract the base model name and try that
            base_model = model.split(':')[0] if ':' in model else model
            base_top_k = self.MODEL_TOP_K.get(base_model, self.TOP_K)
        
        # Apply the multiplier and round to the nearest integer
        # Ensure we always return at least 1 result
        adjusted_top_k = max(1, round(base_top_k * self.TOP_K_MULTIPLIER))
        
        # Log the adjustment if multiplier isn't 1.0
        if self.TOP_K_MULTIPLIER != 1.0:
            logger.info(f"Adjusted top_k for {model} from {base_top_k} to {adjusted_top_k} (multiplier: {self.TOP_K_MULTIPLIER})")
    
        return adjusted_top_k
    
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
        """Delete an archived conversation.
        
        Args:
            username: The username
            archive_name: Name of the archive to delete
            
        Returns:
            Tuple of (success, message)
        """
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
            
            
class UserPreferencesManager:
    """Manages user preferences such as preferred models."""
    
    def __init__(self, base_dir: str = "user_preferences"):
        self.preferences_dir = base_dir
        os.makedirs(self.preferences_dir, exist_ok=True)
    
    def get_file_path(self, user_id: str) -> str:
        """Generate sanitized file path for user preferences."""
        return os.path.join(self.preferences_dir, f"{user_id}.json")
    
    def get_preferred_model(self, user_id: str, default_model: str = "google/gemini-2.0-flash-001") -> str:
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
        
        # Add search caching
        self.search_cache = {}  # Store previous search results by user
        
        # Pass the TOP_K value to DocumentManager
        self.document_manager = DocumentManager(top_k=self.config.TOP_K, config=self.config)

        # Clean up any empty documents - THIS IS THE NEW CODE
        empty_docs = self.document_manager.cleanup_empty_documents()
        if empty_docs:
            logger.info(f"Cleaned up {len(empty_docs)} empty documents at startup")
        
        # Initialize the ImageManager with a reference to the DocumentManager
        self.image_manager = ImageManager(document_manager=self.document_manager)
        
        self.user_preferences_manager = UserPreferencesManager()
        
        self.timeout_duration = 500

        self.banned_users = set()
        self.banned_users_file = 'banned_users.json'
        self.load_banned_users()

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.refresh_google_docs_wrapper, 'interval', hours=6)
        self.scheduler.start()
        
        # List of models that support vision capabilities
        self.vision_capable_models = [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-pro-exp-02-05:free",
            "anthropic/claude-3.7-sonnet:beta",
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.5-sonnet:beta", 
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku:beta",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-haiku:beta"
        ]

    def sanitize_discord_text(self, text: str) -> str:
        """Sanitize text for Discord message display by escaping special characters."""
        # Replace backticks with single quotes to avoid breaking code blocks
        #text = text.replace("`", "'")
        # Replace backslashes to avoid escape sequence issues
        text = text.replace("\\", "\\\\")
        return text
    
    def refresh_google_docs_wrapper(self):
        """Wrapper to run the async refresh_google_docs method."""
        asyncio.create_task(self.refresh_google_docs())

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
                "model": "google/gemini-2.0-flash-001",  # Use a vision-capable model
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
        return self.document_manager.search(query, top_k=top_k)
                
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
            
        search_results = self.document_manager.search(query, top_k=top_k)
        
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
                    file_name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                    if not file_name.endswith('.txt'):
                        file_name += '.txt'
                    
                    # Check if document has changed
                    changed = await self._has_google_doc_changed(doc_id, file_name)
                    
                    if changed:
                        logger.info(f"Google Doc {doc_id} has changed, updating")
                        
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
                            if file_name in self.document_manager.contextualized_chunks:
                                del self.document_manager.contextualized_chunks[file_name]
                            if file_name in self.document_manager.bm25_indexes:
                                del self.document_manager.bm25_indexes[file_name]
                            
                        # Add document without saving to disk yet
                        await self.document_manager.add_document(file_name, content, save_to_disk=False)
                        updated_docs = True
                    else:
                        logger.info(f"Google Doc {doc_id} has not changed, skipping")
                        
                except Exception as e:
                    logger.error(f"Error refreshing doc {doc_id}: {e}")
            
            # Save to disk once at the end if any docs were updated
            if updated_docs:
                self.document_manager._save_to_disk()
        except Exception as e:
            logger.error(f"Error refreshing Google Docs: {e}")

    async def _has_google_doc_changed(self, doc_id: str, file_name: str) -> bool:
        """Check if a Google Doc has changed since last refresh using content hashing.
        
        Args:
            doc_id: The Google Doc ID
            file_name: The file name used to store the document
            
        Returns:
            bool: True if the document has changed, False otherwise
        """
        try:
            # Get the stored hash if available
            stored_hash = None
            doc_metadata_path = self.document_manager.base_dir / f"{file_name}.metadata.json"
            
            if doc_metadata_path.exists():
                with open(doc_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    stored_hash = metadata.get('content_hash')
            
            # Get the current document content
            file_path = self.document_manager.base_dir / file_name
            if not file_path.exists():
                # If the file doesn't exist locally, it needs to be downloaded
                return True
            
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
                
            # Compute hash of current content
            import hashlib
            current_hash = hashlib.md5(current_content.encode('utf-8')).hexdigest()
            
            # Download the latest version
            async with aiohttp.ClientSession() as session:
                url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
                async with session.get(url) as response:
                    if response.status != 200:
                        # If we can't download, assume document has changed
                        logger.warning(f"Failed to download {doc_id}, assuming document has changed")
                        return True
                    
                    new_content = await response.text()
            
            # Compute hash of new content
            new_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
            
            # Save the new hash
            metadata = {
                'content_hash': new_hash
            }
            with open(doc_metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Compare hashes
            if stored_hash is not None and stored_hash == new_hash:
                return False
            
            # If the content has actually changed
            if current_hash != new_hash:
                return True
            else:
                # Content hasn't changed
                return False
                    
        except Exception as e:
            logger.error(f"Error checking if Google Doc {doc_id} has changed: {e}")
            # If there's an error, assume document has changed
            return True
            
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
            
            # Check if document has changed
            changed = await self._has_google_doc_changed(doc_id, file_name)
            
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
                if old_filename in self.document_manager.contextualized_chunks:
                    del self.document_manager.contextualized_chunks[old_filename]
                if old_filename in self.document_manager.bm25_indexes:
                    del self.document_manager.bm25_indexes[old_filename]
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
                if file_name in self.document_manager.contextualized_chunks:
                    del self.document_manager.contextualized_chunks[file_name]
                if file_name in self.document_manager.bm25_indexes:
                    del self.document_manager.bm25_indexes[file_name]
                
            # Add document and save to disk
            await self.document_manager.add_document(file_name, content)
            
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
        """Set up all slash and prefix commands."""
        
        # Add this as a traditional prefix command instead of slash command
        @self.command(name="add_doc", brief="Add a new document to the knowledge base. Usage: Publicia! add_doc \"Document Name\"")
        async def adddoc_prefix(ctx, *, args):
            """Add a document via prefix command with optional file attachment."""
            try:
                # Extract name from quotation marks
                match = re.match(r'"([^"]+)"', args)
                
                if not match:
                    await ctx.send('*neural error detected!* Please provide a name in quotes. Example: `Publicia! add_doc "Document Name"`')
                    return
                    
                name = match.group(1)  # The text between quotes
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

        @self.command(name="add_image", brief="Add an image to the knowledge base. \nUsage: `Publicia! add_image \"Your Image Name\" [yes/no]` \n(yes/no controls whether to auto-generate a description, default is yes)")
        async def addimage_prefix(ctx, *, args=""):
            """Add an image via prefix command with file attachment.
            
            Usage:
            - Publicia! add_image "Your Image Name" [yes/no]
            (yes/no controls whether to auto-generate a description, default is yes)
            """
            try:
                # Parse arguments to extract name and generate_description option
                match = re.match(r'"([^"]+)"\s*(\w*)', args)
                
                if not match:
                    await ctx.send('*neural error detected!* Please provide a name in quotes. Example: `Publicia! add_image "Image Name" yes`')
                    return
                    
                name = match.group(1)  # The text between quotes
                generate_description = match.group(2).lower() or "yes"  # The word after the quotes, default to "yes"
                
                # Check for attachments
                if not ctx.message.attachments:
                    await ctx.send("*neural error detected!* Please attach an image to your message.")
                    return
                    
                # Process the first image attachment
                valid_attachment = None
                for attachment in ctx.message.attachments:
                    if is_image(attachment):
                        valid_attachment = attachment
                        break
                        
                if not valid_attachment:
                    await ctx.send("*neural error detected!* No valid image attachment found. Please make sure you're attaching an image file.")
                    return
                    
                # Download image
                async with aiohttp.ClientSession() as session:
                    async with session.get(valid_attachment.url) as resp:
                        if resp.status != 200:
                            await ctx.send(f"*neural error detected!* Failed to download image (status: {resp.status})")
                            return
                        image_data = await resp.read()
                
                # Status message
                status_msg = await ctx.send("*neural pathways activating... processing image...*")
                
                # Handle description based on user choice
                if generate_description == "yes":
                    await status_msg.edit(content="*neural pathways activating... analyzing image content...*")
                    description = await self._generate_image_description(image_data)
                    if description == "Error generating description.":
                        await ctx.send("*neural circuit overload!* An error occurred while processing the image.")
                        return
                    description = name + ": " + description
                    
                    # Add to image manager
                    image_id = self.image_manager.add_image(name, image_data, description)
                    
                    # Success message with preview of auto-generated description
                    description_preview = description[:1000] + "..." if len(description) > 1000 else description
                    success_message = f"*neural analysis complete!* Added image '{name}' to my knowledge base with ID: {image_id}\n\nGenerated description: {description_preview}"
                    await status_msg.edit(content=success_message)
                else:
                    # Ask user to provide a description
                    await status_msg.edit(content="Please provide a description for the image (type it and send within 60 seconds):")
                    
                    try:
                        # Wait for user to type description
                        description_msg = await self.wait_for(
                            'message',
                            timeout=60.0,
                            check=lambda m: m.author == ctx.author and m.channel == ctx.channel
                        )
                        description = description_msg.content
                        
                        # Add to image manager
                        image_id = self.image_manager.add_image(name, image_data, description)
                        
                        await ctx.send(f"*neural pathways reconfigured!* Added image '{name}' with your custom description to my knowledge base with ID: {image_id}")
                    except asyncio.TimeoutError:
                        await status_msg.edit(content="*neural pathway timeout!* You didn't provide a description within the time limit.")
                        return
                    
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                await ctx.send("*neural circuit overload!* An error occurred while processing the image.")

        @self.tree.command(name="add_info", description="Add new text to Publicia's mind for retrieval")
        @app_commands.describe(
            name="Name of the document",
            content="Content of the document"
        )
        async def add_document(interaction: discord.Interaction, name: str, content: str):
            await interaction.response.defer()
            try:
                if not name or not content:
                    await interaction.followup.send("*neural error detected!* Both name and content are required.")
                    return
                    
                lorebooks_path = self.document_manager.get_lorebooks_path()
                txt_path = lorebooks_path / f"{name}.txt"
                txt_path.write_text(content, encoding='utf-8')
                
                self.document_manager.add_document(name, content)
                await interaction.followup.send(f"Added document: {name}\nSaved to: {txt_path}")
                
            except Exception as e:
                logger.error(f"Error adding document: {e}")
                await interaction.followup.send(f"Error adding document: {str(e)}")

        @self.tree.command(name="manage_history", description="View and manage your conversation history")
        @app_commands.describe(limit="Number of messages to display (default: 10, max: 50)")
        async def manage_history(interaction: discord.Interaction, limit: int = 10):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                # Validate limit
                if limit <= 0:
                    await interaction.followup.send("*neural error detected!* The limit must be a positive number.")
                    return
                
                # Cap limit at 50 to prevent excessive output
                limit = min(limit, 50)
                
                # Get limited conversation history
                recent_messages = self.conversation_manager.get_limited_history(interaction.user.name, limit)
                
                if not recent_messages:
                    await interaction.followup.send("*neural pathways empty!* I don't have any conversation history with you yet.")
                    return
                
                # Format conversation history
                response = "*accessing neural memory banks...*\n\n"
                response += f"**CONVERSATION HISTORY** (showing last {len(recent_messages)} messages)\n\n"
                
                # Format each message
                for msg in recent_messages:
                    display_index = msg.get('display_index', 0)
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
                    
                    # Add message to response with index
                    response += f"**[{display_index}]** "
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
                
                # Add instructions for deletion
                response += "*end of neural memory retrieval*\n\n"
                response += "**To delete messages:** Use the `/delete_history_messages` command with these options:\n"
                response += "- `indices`: Comma-separated list of message indices to delete (e.g., '0,2,5')\n"
                response += "- `confirm`: Set to 'yes' to confirm deletion\n\n"
                response += "Example: `/delete_history_messages indices:1,3 confirm:yes` will delete messages [1] and [3] from what you see above."
                
                # Send the response, splitting if necessary
                for chunk in split_message(response):
                    await interaction.followup.send(chunk, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error managing conversation history: {e}")
                await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to manage your conversation history.")

        @self.tree.command(name="delete_history_messages", description="Delete specific messages from your conversation history")
        @app_commands.describe(
            indices="Comma-separated list of message indices to delete (e.g., '0,2,5')",
            confirm="Type 'yes' to confirm deletion"
        )
        async def delete_history_messages(
            interaction: discord.Interaction, 
            indices: str,
            confirm: str = "no"
        ):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                # Check confirmation
                if confirm.lower() != "yes":
                    await interaction.followup.send("*deletion aborted!* Please confirm by setting `confirm` to 'yes'.")
                    return
                    
                # Parse indices
                try:
                    indices_list = [int(idx.strip()) for idx in indices.split(',') if idx.strip()]
                    if not indices_list:
                        await interaction.followup.send("*neural error detected!* Please provide valid message indices as a comma-separated list (e.g., '0,2,5').")
                        return
                except ValueError:
                    await interaction.followup.send("*neural error detected!* Please provide valid integer indices.")
                    return
                    
                # Delete messages
                success, message, deleted_count = self.conversation_manager.delete_messages_by_display_index(
                    interaction.user.name, 
                    indices_list,
                    limit=50  # Use same max limit as manage_history
                )
                
                if success:
                    if deleted_count > 0:
                        await interaction.followup.send(f"*neural pathways reconfigured!* {message}")
                    else:
                        await interaction.followup.send("*no changes made!* No messages were deleted.")
                else:
                    await interaction.followup.send(f"*neural error detected!* {message}")
                    
            except Exception as e:
                logger.error(f"Error deleting messages: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while deleting messages.")

        @self.tree.command(name="archive_conversation", description="Archive your current conversation history")
        @app_commands.describe(
            archive_name="Optional name for the archive (defaults to timestamp)"
        )
        async def archive_conversation(interaction: discord.Interaction, archive_name: str = None):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                success, message = self.conversation_manager.archive_conversation(interaction.user.name, archive_name)
                
                if success:
                    await interaction.followup.send(f"*neural storage complete!* {message}", ephemeral=True)
                else:
                    await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error archiving conversation: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while archiving the conversation.", ephemeral=True)

        @self.tree.command(name="list_archives", description="List your archived conversation histories")
        async def list_archives(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                archives = self.conversation_manager.list_archives(interaction.user.name)
                
                if not archives:
                    await interaction.followup.send("*neural archives empty!* You don't have any archived conversations.", ephemeral=True)
                    return
                
                # Format the list of archives
                response = "*accessing neural storage banks...*\n\n"
                response += "**ARCHIVED CONVERSATIONS**\n\n"
                
                for i, archive in enumerate(archives):
                    # Try to extract a timestamp from the filename
                    timestamp_match = re.search(r'(\d{8}_\d{6})', archive)
                    if timestamp_match:
                        try:
                            timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
                            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            response += f"{i+1}. **{archive}** (Created: {formatted_time})\n"
                        except:
                            response += f"{i+1}. **{archive}**\n"
                    else:
                        response += f"{i+1}. **{archive}**\n"
                
                response += "\n*use `/swap_conversation` to swap between current and archived conversations*"
                
                await interaction.followup.send(response, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error listing archives: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while listing archives.", ephemeral=True)

        @self.tree.command(name="swap_conversation", description="Swap between current and archived conversation histories")
        @app_commands.describe(
            archive_name="Name of the archive to swap with"
        )
        async def swap_conversation(interaction: discord.Interaction, archive_name: str):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                success, message = self.conversation_manager.swap_conversation(interaction.user.name, archive_name)
                
                if success:
                    await interaction.followup.send(f"*neural reconfiguration complete!* {message}", ephemeral=True)
                else:
                    await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error swapping conversation: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while swapping conversations.", ephemeral=True)

        @self.tree.command(name="delete_archive", description="Delete an archived conversation")
        @app_commands.describe(
            archive_name="Name of the archive to delete",
            confirm="Type 'yes' to confirm deletion"
        )
        async def delete_archive(interaction: discord.Interaction, archive_name: str, confirm: str = "no"):
            await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
            try:
                # Check confirmation
                if confirm.lower() != "yes":
                    await interaction.followup.send("*deletion aborted!* Please confirm by setting `confirm` to 'yes'.", ephemeral=True)
                    return
                    
                success, message = self.conversation_manager.delete_archive(interaction.user.name, archive_name)
                
                if success:
                    await interaction.followup.send(f"*neural memory purged!* {message}", ephemeral=True)
                else:
                    # If archive not found, list available archives
                    if "not found" in message:
                        archives = self.conversation_manager.list_archives(interaction.user.name)
                        if archives:
                            archives_list = "\n".join([f"• {archive}" for archive in archives])
                            await interaction.followup.send(f"*neural error detected!* {message}\n\nAvailable archives:\n{archives_list}", ephemeral=True)
                        else:
                            await interaction.followup.send(f"*neural error detected!* {message}\nYou don't have any archived conversations.", ephemeral=True)
                    else:
                        await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error deleting archive: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while deleting the archive.", ephemeral=True)

        @self.tree.command(name="compare_models", description="Compare responses from multiple AI models (admin only)")
        @app_commands.describe(
            question="Question to ask all models",
            model_types="Types of models to include, comma-separated (defaults to all)",
            max_models="Maximum number of models to test (0 for unlimited)",
            image_url="Optional URL to an image for vision-capable models",
            private="Make the results visible only to you (default: True)"
        )
        @app_commands.check(check_permissions)
        async def compare_models(
            interaction: discord.Interaction, 
            question: str, 
            model_types: str = "all", 
            max_models: int = 0, 
            image_url: str = None,
            private: bool = True
        ):
            await interaction.response.defer(ephemeral=private)
            try:
                if not question:
                    await interaction.followup.send("*neural error detected!* Please provide a question.", ephemeral=private)
                    return
                    
                # Parse model types
                requested_types = [t.strip().lower() for t in model_types.split(',')]
                include_all = "all" in requested_types
                
                # Send initial status message
                status_message = await interaction.followup.send(
                    "*neural pathways activating... preparing to test models...*",
                    ephemeral=private
                )
                
                # Process image URL if provided
                image_attachments = []
                if image_url:
                    try:
                        # Check if URL appears to be a direct image link
                        if any(image_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            await status_message.edit(content="*neural pathways activating... downloading image...*")
                            
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
                                            image_attachments.append(f"data:{content_type};base64,{base64_data}")
                                            logger.info(f"Processed image from URL: {image_url}")
                                        else:
                                            await interaction.followup.send("*neural error detected!* The URL does not point to a valid image.", ephemeral=private)
                                            return
                                    else:
                                        await interaction.followup.send(f"*neural error detected!* Could not download image (status code: {resp.status}).", ephemeral=private)
                                        return
                        else:
                            await interaction.followup.send("*neural error detected!* The URL does not appear to be a direct image link. Please provide a URL ending with .jpg, .png, etc.", ephemeral=private)
                            return
                    except Exception as e:
                        logger.error(f"Error processing image URL: {e}")
                        await interaction.followup.send("*neural error detected!* Failed to process the image URL.", ephemeral=private)
                        return
                
                # Collect models based on requested types
                models_to_try = []
                
                # Define model categories
                model_categories = {
                    "deepseek": [
                        "deepseek/deepseek-r1",
                    ],
                    "google": [
                        "google/gemini-2.0-flash-001",
                    ],
                    "claude": [
                        "anthropic/claude-3.5-haiku:beta",
                        "anthropic/claude-3.5-sonnet:beta",
                        "anthropic/claude-3.7-sonnet:beta",
                    ],
                    "qwen": [
                        "qwen/qwq-32b",
                    ],
                    "nous": [
                        "nousresearch/hermes-3-llama-3.1-405b",
                    ],
                    "testing": [
                        "eva-unit-01/eva-qwen-2.5-72b",
                        "thedrummer/unslopnemo-12b",
                    ],
                    "storytelling": [
                        "latitudegames/wayfarer-large-70b-llama-3.3",
                        "thedrummer/anubis-pro-105b-v1"
                    ],
                }
                
                # Add models based on requested types
                for category, models in model_categories.items():
                    if include_all or category in requested_types or any(req_type in category for req_type in requested_types):
                        for model in models:
                            if model not in models_to_try:
                                models_to_try.append(model)
                
                # If no valid types specified, use all
                if not models_to_try:
                    for models in model_categories.values():
                        for model in models:
                            if model not in models_to_try:
                                models_to_try.append(model)
                                
                # Limit number of models if requested
                if max_models > 0 and len(models_to_try) > max_models:
                    models_to_try = models_to_try[:max_models]
                    
                # Update status
                await status_message.edit(content=f"*neural pathways activating... testing {len(models_to_try)} models...*")
                
                # Prepare messages for the models
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
                
                # Dictionary to store results
                results = {}
                
                # Try each model and collect results
                for i, model in enumerate(models_to_try):
                    try:
                        await status_message.edit(content=f"*neural pathways testing... model {i+1}/{len(models_to_try)}: {model}*")
                        
                        # Check if this is a vision model and we have image attachments
                        is_vision_model = model in self.vision_capable_models
                        use_images = is_vision_model and image_attachments
                        
                        # Use the try_ai_completion method
                        completion, actual_model = await self._try_ai_completion(
                            model,
                            messages,
                            image_attachments=image_attachments if use_images else None,
                            temperature=0.1
                        )
                        
                        if completion and completion.get('choices') and len(completion['choices']) > 0:
                            if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                                response = completion['choices'][0]['message']['content']
                                results[model] = {
                                    "actual_model": actual_model or model,
                                    "response": response,
                                    "status": "Success",
                                    "is_vision_model": is_vision_model,
                                    "used_image": use_images
                                }
                            else:
                                results[model] = {
                                    "actual_model": actual_model or model,
                                    "response": "Error: Unexpected response structure",
                                    "status": "Error",
                                    "is_vision_model": is_vision_model,
                                    "used_image": use_images
                                }
                        else:
                            results[model] = {
                                "actual_model": actual_model or "None",
                                "response": "Error: No response received",
                                "status": "Error",
                                "is_vision_model": is_vision_model,
                                "used_image": use_images
                            }
                            
                    except Exception as e:
                        results[model] = {
                            "actual_model": model,
                            "response": f"Error: {str(e)}",
                            "status": "Exception",
                            "is_vision_model": model in self.vision_capable_models,
                            "used_image": False
                        }
                        logger.error(f"Error with model {model}: {e}")
                        
                # Create a text file with the results
                file_content = f"""
        =========================================================
        PUBLICIA MODEL COMPARISON RESULTS
        =========================================================
        Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Query: {question}
        Models tested: {len(models_to_try)}
        Timeout per model: {self.config.API_TIMEOUT} seconds (from config)
        Image included: {"Yes" if image_url else "No"}

        This file contains responses from multiple AI models to the same query,
        allowing for comparison of how different models interpret and respond
        to the same prompt.
        =========================================================

        """
                
                # Add a summary of successful/failed models
                success_count = sum(1 for r in results.values() if r["status"] == "Success")
                error_count = sum(1 for r in results.values() if r["status"] == "Error" or r["status"] == "Exception")
                vision_count = sum(1 for r in results.values() if r["is_vision_model"])
                
                file_content += f"""
        SUMMARY:
        ---------------------------------------------------------
        Total models: {len(models_to_try)}
        Successful: {success_count}
        Errors: {error_count}
        Vision-capable models: {vision_count}
        =========================================================

        """
                
                # Add responses from each model
                for model, result in results.items():
                    actual_model = result["actual_model"]
                    response = result["response"]
                    status = result["status"]
                    is_vision = result["is_vision_model"]
                    used_image = result["used_image"]
                    
                    file_content += f"""
        MODEL: {model}
        STATUS: {status}
        ACTUAL MODEL USED: {actual_model}
        VISION-CAPABLE: {"Yes" if is_vision else "No"}
        IMAGE USED: {"Yes" if used_image else "No"}
        ---------------------------------------------------------
        {response}
        =========================================================

        """
                
                # Save to file
                file_name = f"publicia_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(file_content)
                    
                # Check file size and truncate if necessary
                file_size = os.path.getsize(file_name)
                if file_size > 8 * 1024 * 1024:  # 8MB Discord limit
                    truncated_file_name = f"publicia_model_comparison_truncated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(file_name, "r", encoding="utf-8") as f_in:
                        with open(truncated_file_name, "w", encoding="utf-8") as f_out:
                            f_out.write(f"WARNING: Original file was {file_size / (1024*1024):.2f}MB, exceeding Discord's limit. Content has been truncated.\n\n")
                            f_out.write(f_in.read(7 * 1024 * 1024))
                            f_out.write("\n\n[... Content truncated due to file size limits ...]")
                            
                    os.remove(file_name)
                    file_name = truncated_file_name
                
                # Upload file
                await status_message.edit(content="*uploading model comparison results...*")
                await interaction.followup.send(
                    file=discord.File(file_name), 
                    content=f"*here are the responses from {len(results)} models to your query:*",
                    ephemeral=private
                )
                
                # Clean up
                os.remove(file_name)
                await status_message.edit(content="*model comparison complete!*")
                
            except Exception as e:
                logger.error(f"Error in compare_models command: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await interaction.followup.send("*neural circuit overload!* failed to complete model comparison due to an error.", ephemeral=private)

        @self.tree.command(name="export_prompt", description="Export the full prompt that would be sent to the AI for your query")
        @app_commands.describe(
            question="The question to generate a prompt for",
            private="Whether to make the output visible only to you (default: True)"
        )
        async def export_prompt(interaction: discord.Interaction, question: str, private: bool = True):
            """Export the complete prompt that would be sent to the AI model."""
            await interaction.response.defer(ephemeral=private)
            try:
                # This command handles just like a regular query, but exports the prompt instead
                # of sending it to the AI model
                
                status_message = await interaction.followup.send(
                    "*neural pathways activating... processing query for prompt export...*",
                    ephemeral=private
                )
                
                # Get channel and user info
                channel_name = interaction.channel.name if interaction.guild else "DM"
                nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name
                
                # Process exactly like a regular query until we have the messages
                conversation_messages = self.conversation_manager.get_conversation_messages(interaction.user.name)
                
                await status_message.edit(content="*analyzing query...*")
                analysis = await self.analyze_query(question)
                
                await status_message.edit(content="*searching imperial databases...*")
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )
                search_results = await self.enhanced_search(question, analysis, preferred_model)
                
                await status_message.edit(content="*synthesizing information...*")
                synthesis = await self.synthesize_results(question, search_results, analysis)
                
                # Now we have all the necessary information to create the prompt
                
                # Format the prompt for export
                await status_message.edit(content="*formatting prompt for export...*")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file_content = f"""
        =========================================================
        PUBLICIA PROMPT EXPORT WITH CONTEXTUAL INFORMATION
        =========================================================
        Generated at: {timestamp}
        Query: {question}
        User: {nickname}
        Channel: {channel_name}
        Preferred model: {preferred_model}

        This file contains the complete prompt that would be sent to 
        the AI model when processing your query. This provides insight
        into how Publicia formulates responses by showing:

        1. The system prompt that defines Publicia's character
        2. Your conversation history for context
        3. Search results from relevant documents
        4. How your query was analyzed
        5. The final message containing your question

        SPECIAL FEATURE: This export shows whether document chunks have
        been enhanced with contextual information to improve search quality.

        NOTE: This export includes your conversation history, which
        may contain personal information. If you're sharing this file,
        please review the content first.
        =========================================================

        """
                
                # System prompt
                file_content += f"""
        SYSTEM PROMPT
        ---------------------------------------------------------
        This defines Publicia's character, abilities, and behavior.
        ---------------------------------------------------------
        {SYSTEM_PROMPT}
        =========================================================

        """
                
                # Conversation history
                if conversation_messages:
                    file_content += f"""
        CONVERSATION HISTORY ({len(conversation_messages)} messages)
        ---------------------------------------------------------
        Previous messages provide context for your current query.
        ---------------------------------------------------------
        """
                    for i, msg in enumerate(conversation_messages):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        file_content += f"[{i+1}] {role.upper()}: {content}\n\n"
                    
                    file_content += "=========================================================\n\n"
                
                # Synthesized context
                if synthesis:
                    file_content += f"""
        SYNTHESIZED DOCUMENT CONTEXT
        ---------------------------------------------------------
        This is an AI-generated summary of the search results.
        ---------------------------------------------------------
        {synthesis}
        =========================================================

        """
                
                # Raw search results
                if search_results:
                    file_content += f"""
        RAW SEARCH RESULTS ({len(search_results)} results)
        ---------------------------------------------------------
        These are the actual document chunks found by semantic search.
        Each result shows whether it has been enhanced with AI-generated context.
        ---------------------------------------------------------
        """
                    for i, (doc, chunk, score, image_id, chunk_index, total_chunks) in enumerate(search_results):                        
                        # Check if this chunk has been contextualized
                        has_context = False
                        original_chunk = chunk
                        context_part = ""

                        # Check if we have contextualized chunks for this document
                        if hasattr(self.document_manager, 'contextualized_chunks') and doc in self.document_manager.contextualized_chunks:
                            # Check if we have enough contextualized chunks
                            if chunk_index - 1 < len(self.document_manager.contextualized_chunks[doc]):
                                contextualized_chunk = self.document_manager.contextualized_chunks[doc][chunk_index - 1]
                                
                                # Check if this is actually different from the original
                                if contextualized_chunk != original_chunk and original_chunk in contextualized_chunk:
                                    has_context = True
                                    # Try to extract just the context part (assuming it's prepended)
                                    context_start_pos = contextualized_chunk.find(original_chunk)
                                    if context_start_pos > 0:
                                        context_part = contextualized_chunk[:context_start_pos].strip()
                        
                        if image_id:
                            # Image result
                            image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                            file_content += f"[{i+1}] IMAGE: {image_name} (ID: {image_id}, score: {score:.2f})\n"
                            file_content += f"Description: {chunk}\n"
                        else:
                            # Document result
                            file_content += f"[{i+1}] DOCUMENT: {doc} (score: {score:.2f}, chunk {chunk_index}/{total_chunks})\n"
                            file_content += f"Content: {chunk}\n"
                        
                        # Display context information
                        if has_context:
                            file_content += f"[CONTEXTUAL ENHANCEMENT]: YES\n"
                            file_content += f"Added context: {context_part}\n\n"
                        else:
                            file_content += f"[CONTEXTUAL ENHANCEMENT]: NO\n\n"
                    
                    file_content += "=========================================================\n\n"
                
                # User query
                file_content += f"""
        USER QUERY
        ---------------------------------------------------------
        This is your actual question/message sent to Publicia.
        ---------------------------------------------------------
        {nickname}: {question}
        =========================================================

        """
                
                # Analysis data
                if analysis and analysis.get("success"):
                    file_content += f"""
        QUERY ANALYSIS
        ---------------------------------------------------------
        This shows how your query was analyzed to improve search results.
        ---------------------------------------------------------
        {json.dumps(analysis, indent=2)}
        =========================================================
        """
                
                # Save to file
                file_name = f"publicia_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(file_content)
                
                # Check size and truncate if needed
                file_size = os.path.getsize(file_name)
                if file_size > 8 * 1024 * 1024:  # 8MB Discord limit
                    truncated_file_name = f"publicia_prompt_truncated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(file_name, "r", encoding="utf-8") as f_in:
                        with open(truncated_file_name, "w", encoding="utf-8") as f_out:
                            f_out.write(f"WARNING: Original prompt was {file_size / (1024*1024):.2f}MB, exceeding Discord's limit. Content has been truncated.\n\n")
                            f_out.write(f_in.read(7 * 1024 * 1024))
                            f_out.write("\n\n[... Content truncated due to file size limits ...]")
                            
                        os.remove(file_name)
                        file_name = truncated_file_name
                
                # Upload file
                await status_message.edit(content="*uploading prompt file...*")
                await interaction.followup.send(
                    file=discord.File(file_name), 
                    content=f"*here's the full prompt that would be sent to the AI model for your query. this includes the system prompt, conversation history, and search results with contextual enhancement info:*",
                    ephemeral=private
                )
                
                # Clean up
                os.remove(file_name)
                await status_message.edit(content="*prompt export complete!*")
                
            except Exception as e:
                logger.error(f"Error exporting prompt: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await interaction.followup.send("*neural circuit overload!* failed to export prompt due to an error.")
        
        @self.tree.command(name="list_commands", description="List all available commands")
        async def list_commands(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                response = "*accessing command database through enhanced synapses...*\n\n"
                response += "**AVAILABLE COMMANDS**\n\n"
                
                # SLASH COMMANDS SECTION
                response += "**Slash Commands** (`/command`)\n\n"
                
                categories = {
                    "Lore Queries": ["query"],
                    "Document Management": ["add_info", "list_docs", "remove_doc", "search_docs", "add_googledoc", "list_googledocs", "remove_googledoc", "rename_document", "list_files", "retrieve_file"],
                    "Image Management": ["list_images", "view_image", "edit_image", "remove_image", "update_image_description"],
                    "Utility": ["list_commands", "set_model", "get_model", "toggle_debug", "help", "export_prompt", "reload_docs"],
                    "Memory Management": ["lobotomise", "history", "manage_history", "delete_history_messages", "parse_channel", "archive_conversation", "list_archives", "swap_conversation", "delete_archive"], 
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
                
                # PREFIX COMMANDS SECTION
                response += "**Prefix Commands** (`Publicia! command`)\n\n"
                
                # Get prefix commands from the bot
                prefix_commands = sorted(self.commands, key=lambda x: x.name)
                
                # Group prefix commands by category (estimate categories based on names)
                prefix_categories = {
                    "Document Management": [],
                    "Image Management": [],
                }
                
                # Sort commands into categories
                for cmd in prefix_commands:
                    if "doc" in cmd.name.lower():
                        prefix_categories["Document Management"].append(cmd)
                    elif "image" in cmd.name.lower():
                        prefix_categories["Image Management"].append(cmd)
                
                # Format and add each category of prefix commands
                for category, cmds in prefix_categories.items():
                    if cmds:  # Only show categories that have commands
                        response += f"__*{category}*__\n"
                        for cmd in cmds:
                            brief = cmd.brief or "No description available"
                            response += f"`Publicia! {cmd.name}`: {brief}\n"
                        response += "\n"
                
                response += "\n*you can ask questions about ledus banum 77 and imperial lore by pinging/mentioning me or using the /query command!*"
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
            await interaction.response.defer(ephemeral=True)
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
                    await interaction.followup.send(chunk, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error displaying conversation history: {e}")
                await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to retrieve your conversation history.")
        
        @self.tree.command(name="set_model", description="Set your preferred AI model for responses")
        @app_commands.describe(model="Choose the AI model you prefer")
        @app_commands.choices(model=[
            app_commands.Choice(name="DeepSeek-R1", value="deepseek/deepseek-r1:free"),
            app_commands.Choice(name="Gemini 2.0 Flash", value="google/gemini-2.0-flash-001"),
            app_commands.Choice(name="Nous: Hermes 405B", value="nousresearch/hermes-3-llama-3.1-405b"),
            app_commands.Choice(name="Qwen QwQ 32B", value="qwen/qwq-32b:free"),
            app_commands.Choice(name="Claude 3.5 Haiku", value="anthropic/claude-3.5-haiku:beta"),
            app_commands.Choice(name="Claude 3.5 Sonnet", value="anthropic/claude-3.5-sonnet:beta"),
            app_commands.Choice(name="Claude 3.7 Sonnet", value="anthropic/claude-3.7-sonnet:beta"),
            app_commands.Choice(name="Testing Model", value="eva-unit-01/eva-qwen-2.5-72b"),
            # Add the new models here
            app_commands.Choice(name="Wayfarer 70B", value="latitudegames/wayfarer-large-70b-llama-3.3"),
            app_commands.Choice(name="Anubis Pro 105B", value="thedrummer/anubis-pro-105b-v1"),
        ])
        async def set_model(interaction: discord.Interaction, model: str):
            await interaction.response.defer()
            try:
                # Check if user is allowed to use Claude 3.7 Sonnet
                if model == "anthropic/claude-3.7-sonnet:beta" and str(interaction.user.id) != "203229662967627777" and not (interaction.guild and interaction.user.guild_permissions.administrator):
                    await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                    return

                if model == "anthropic/claude-3.5-sonnet:beta" and str(interaction.user.id) != "203229662967627777" and not (interaction.guild and interaction.user.guild_permissions.administrator):
                    await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                    return
                    
                success = self.user_preferences_manager.set_preferred_model(str(interaction.user.id), model)
                
                # Get friendly model name based on the model value
                model_name = "Unknown Model"
                if "deepseek/deepseek-r1" in model:
                    model_name = "DeepSeek-R1"
                elif model.startswith("google/"):
                    model_name = "Gemini 2.0 Flash"
                elif model.startswith("nousresearch/"):
                    model_name = "Nous: Hermes 405B Instruct"
                elif "claude-3.5-haiku" in model:
                    model_name = "Claude 3.5 Haiku"
                elif "claude-3.5-sonnet" in model:
                    model_name = "Claude 3.5 Sonnet"
                elif "claude-3.7-sonnet" in model:
                    model_name = "Claude 3.7 Sonnet"
                elif "qwen/qwq-32b" in model:
                    model_name = "Qwen QwQ 32B"
                elif "unslopnemo" in model or "eva-unit-01/eva-qwen-2.5-72b" in model:
                    model_name = "Testing Model"
                elif "latitudegames/wayfarer" in model:
                    model_name = "Wayfarer 70B"
                elif "thedrummer/anubis-pro" in model:
                    model_name = "Anubis Pro 105B"
                
                if success:
                    # Create a description of all model strengths
                    # Create a description of all model strengths
                    model_descriptions = [
                        f"**DeepSeek-R1**: Great for roleplaying, more creative responses, and in-character immersion, but is slower to respond, sometimes has errors, and may make things up due to its creativity. With free version uses ({self.config.get_top_k_for_model('deepseek/deepseek-r1:free')}) search results, otherwise uses ({self.config.get_top_k_for_model('deepseek/deepseek-r1')}).",
                        f"**Gemini 2.0 Flash**: RECOMMENDED - Better for accurate citations, factual responses, document analysis, image viewing capabilities, and has very fast response times. Uses more search results ({self.config.get_top_k_for_model('google/gemini-2.0-flash-001')}) for broader context.",
                        f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({self.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}) for balanced context.",
                        f"**Qwen QwQ 32B**: RECOMMENDED - Great for roleplaying with strong lore accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Prone to minor hallucinations due to it being a small model, and it sometimes slips in Chinese phrases. Uses ({self.config.get_top_k_for_model('qwen/qwq-32b:free')}) with the free model, otherwise uses ({self.config.get_top_k_for_model('qwen/qwq-32b')}).",
                        f"**Claude 3.5 Haiku**: Excellent for comprehensive lore analysis and nuanced understanding with creativity, and has image viewing capabilities. Also great for longer roleplays. Uses a moderate number of search results ({self.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}) for balanced context.",
                        f"**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only). Uses fewer search results ({self.config.get_top_k_for_model('anthropic/claude-3.5-sonnet')}) to save money.",
                        f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({self.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                        f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({self.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                        f"**Wayfarer 70B**: Optimized for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({self.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                        f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({self.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                    ]
                    
                    response = f"*neural architecture reconfigured!* Your preferred model has been set to **{model_name}**.\n\n**Model strengths:**\n"
                    response += "\n".join(model_descriptions)
                    
                    for chunk in split_message(response):
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send("*synaptic error detected!* Failed to set your preferred model. Please try again later.")
                    
            except Exception as e:
                logger.error(f"Error setting preferred model: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while setting your preferred model.")

        @self.tree.command(name="get_model", description="Show your currently selected AI model and available models")
        async def get_model(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )
                
                # Get friendly model name based on the model value
                model_name = "Unknown Model"
                if "deepseek/deepseek-r1" in preferred_model:
                    model_name = "DeepSeek-R1"
                elif preferred_model.startswith("google/"):
                    model_name = "Gemini 2.0 Flash"
                elif preferred_model.startswith("nousresearch/"):
                    model_name = "Nous: Hermes 405B Instruct"
                elif "claude-3.5-haiku" in preferred_model:
                    model_name = "Claude 3.5 Haiku"
                elif "claude-3.5-sonnet" in preferred_model:
                    model_name = "Claude 3.5 Sonnet"
                elif "claude-3.7-sonnet" in preferred_model:
                    model_name = "Claude 3.7 Sonnet"
                elif "qwen/qwq-32b" in preferred_model:
                    model_name = "Qwen QwQ 32B"
                elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                    model_name = "Testing Model"
                elif "latitudegames/wayfarer" in preferred_model:
                    model_name = "Wayfarer 70B"
                elif "thedrummer/anubis-pro" in preferred_model:
                    model_name = "Anubis Pro 105B"
                
                # Create a description of all model strengths
                model_descriptions = [
                    f"**DeepSeek-R1**: Great for roleplaying, more creative responses, and in-character immersion, but is slower to respond, sometimes has errors, and may make things up due to its creativity. With free version uses ({self.config.get_top_k_for_model('deepseek/deepseek-r1:free')}) search results, otherwise uses ({self.config.get_top_k_for_model('deepseek/deepseek-r1')}).",
                    f"**Gemini 2.0 Flash**: RECOMMENDED - Better for accurate citations, factual responses, document analysis, image viewing capabilities, and has very fast response times. Uses more search results ({self.config.get_top_k_for_model('google/gemini-2.0-flash-001')}) for broader context.",
                    f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({self.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}) for balanced context.",
                    f"**Qwen QwQ 32B**: RECOMMENDED - Great for roleplaying with strong lore accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Prone to minor hallucinations due to it being a small model, and it sometimes slips in Chinese phrases. Uses ({self.config.get_top_k_for_model('qwen/qwq-32b:free')}) search results.",
                    f"**Claude 3.5 Haiku**: Excellent for comprehensive lore analysis and nuanced understanding with creativity, and has image viewing capabilities. Also great for longer roleplays. Uses a moderate number of search results ({self.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}) for balanced context.",
                    f"**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only). Uses fewer search results ({self.config.get_top_k_for_model('anthropic/claude-3.5-sonnet')}) to save money.",
                    f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({self.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                    f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({self.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                    f"**Wayfarer 70B**: Optimized for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({self.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                    f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({self.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                ]
                
                response = f"*neural architecture scan complete!* Your currently selected model is **{model_name}**.\n\n**Model strengths:**\n"
                response += "\n".join(model_descriptions)
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
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
                

        @self.tree.command(name="list_docs", description="List all available documents")
        async def list_documents(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                if not self.document_manager.metadata:
                    await interaction.followup.send("No documents found in the knowledge base.")
                    return
                    
                # Get all documents first
                doc_items = []
                for doc_name, meta in self.document_manager.metadata.items():
                    chunks = meta['chunk_count']
                    added = meta['added']
                    doc_items.append(f"{doc_name} - {chunks} chunks (Added: {added})")
                
                # Create header
                header = "Available documents:"
                
                # Split into chunks, allowing room for code block formatting
                doc_chunks = split_message("\n".join(doc_items), max_length=1900)  # Leave room for formatting
                
                for i, chunk in enumerate(doc_chunks):
                    # Format each chunk as a separate code block
                    formatted_chunk = f"{header if i == 0 else 'Documents (continued):'}\n```\n{chunk}\n```"
                    await interaction.followup.send(formatted_chunk)
                    
            except Exception as e:
                await interaction.followup.send(f"Error listing documents: {str(e)}")

        @self.tree.command(name="list_images", description="List all images in Publicia's knowledge base")
        async def list_images(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                images = self.image_manager.list_images()
                
                if not images:
                    await interaction.followup.send("*neural pathways empty!* No images found in my knowledge base.")
                    return
                
                response = "*accessing visual memory banks...*\n\n**STORED IMAGES**\n"
                for img in images:
                    added_date = datetime.fromisoformat(img['added']).strftime("%Y-%m-%d %H:%M:%S")
                    response += f"\n**ID**: {img['id']}\n**Name**: {img['name']}\n**Added**: {added_date}\n**Has Description**: {'Yes' if img['has_description'] else 'No'}\n"
                
                # Split the message if necessary
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
            except Exception as e:
                logger.error(f"Error listing images: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while retrieving the image list.")

        @self.tree.command(name="view_image", description="View an image from Publicia's knowledge base")
        @app_commands.describe(image_id="ID of the image to view")
        async def view_image(interaction: discord.Interaction, image_id: str):
            await interaction.response.defer()
            try:
                # Check if image exists
                if image_id not in self.image_manager.metadata:
                    await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                    return
                
                # Get image metadata
                image_meta = self.image_manager.metadata[image_id]
                image_name = image_meta['name']
                image_desc = image_meta.get('description', 'No description available')
                image_path = Path(image_meta['path'])
                
                if not image_path.exists():
                    await interaction.followup.send(f"*neural error detected!* Image file not found for ID: {image_id}")
                    return
                
                # Send description
                description = f"**Image**: {image_name} (ID: {image_id})\n\n**Description**:\n{image_desc}"
                
                # Split if needed
                for chunk in split_message(description):
                    await interaction.followup.send(chunk)
                
                # Send image file
                with open(image_path, 'rb') as f:
                    file = discord.File(f, filename=f"{image_name}.png")
                    await interaction.followup.send(file=file)
                
            except Exception as e:
                logger.error(f"Error viewing image: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while retrieving the image.")

        @self.command(name="edit_image", brief="View and edit an image description. Usage: Publicia! edit_image [image_id]")
        async def edit_image_prefix(ctx, image_id: str):
            """View and edit an image description with a conversational flow."""
            try:
                # Check if image exists
                if image_id not in self.image_manager.metadata:
                    await ctx.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                    return
                
                # Get image metadata
                image_meta = self.image_manager.metadata[image_id]
                image_name = image_meta['name']
                image_desc = image_meta.get('description', 'No description available')
                image_path = Path(image_meta['path'])
                
                if not image_path.exists():
                    await ctx.send(f"*neural error detected!* Image file not found for ID: {image_id}")
                    return
                
                # Send description
                description = f"**Image**: {image_name} (ID: {image_id})\n\n**Current Description**:\n{image_desc}\n\n*To edit this description, reply with a new description within 60 seconds. Type 'cancel' to keep the current description.*"
                
                # Split if needed and send
                for chunk in split_message(description):
                    await ctx.send(chunk)
                
                # Send image file
                with open(image_path, 'rb') as f:
                    file = discord.File(f, filename=f"{image_name}.png")
                    await ctx.send(file=file)
                
                # Wait for the user's response to edit the description
                def check(m):
                    return m.author == ctx.author and m.channel == ctx.channel
                
                try:
                    message = await self.wait_for('message', timeout=60.0, check=check)
                    
                    # Check if user wants to cancel
                    if message.content.lower() == 'cancel':
                        await ctx.send(f"*neural pathway unchanged!* Keeping the current description for image '{image_name}'.")
                        return
                    
                    # Update the description
                    new_description = message.content
                    success = self.image_manager.update_description(image_id, new_description)
                    
                    if success:
                        await ctx.send(f"*neural pathways reconfigured!* Updated description for image '{image_name}'.")
                    else:
                        await ctx.send(f"*neural error detected!* Failed to update description for image '{image_name}'.")
                
                except asyncio.TimeoutError:
                    await ctx.send("*neural pathway timeout!* No description provided within the time limit.")
                    
            except Exception as e:
                logger.error(f"Error editing image description: {e}")
                await ctx.send("*neural circuit overload!* An error occurred while processing the image.")

        @self.tree.command(name="remove_image", description="Remove an image from Publicia's knowledge base")
        @app_commands.describe(image_id="ID of the image to remove")
        async def remove_image(interaction: discord.Interaction, image_id: str):
            await interaction.response.defer()
            try:
                success = self.image_manager.delete_image(image_id)
                
                if success:
                    await interaction.followup.send(f"*neural pathways reconfigured!* Removed image with ID: {image_id}")
                else:
                    await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                    
            except Exception as e:
                logger.error(f"Error removing image: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while removing the image.")

        @self.tree.command(name="update_image_description", description="Update the description for an image")
        @app_commands.describe(
            image_id="ID of the image to update",
            description="New description for the image"
        )
        async def update_image_description(interaction: discord.Interaction, image_id: str, description: str):
            await interaction.response.defer()
            try:
                if not description:
                    await interaction.followup.send("*neural error detected!* Description cannot be empty.")
                    return
                    
                if image_id not in self.image_manager.metadata:
                    await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                    return
                    
                success = self.image_manager.update_description(image_id, description)
                
                if success:
                    await interaction.followup.send(f"*neural pathways reconfigured!* Updated description for image with ID: {image_id}")
                else:
                    await interaction.followup.send(f"*neural error detected!* Could not update image description for ID: {image_id}")
                    
            except Exception as e:
                logger.error(f"Error updating image description: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while updating the image description.")

        @self.tree.command(name="parse_channel", description="Toggle parsing of channel messages for context")
        @app_commands.describe(
            enabled="Enable or disable channel message parsing",
            message_count="Number of messages to retrieve (max 100)"
        )
        async def toggle_channel_parsing(
            interaction: discord.Interaction, 
            enabled: bool, 
            message_count: int = 20
        ):
            await interaction.response.defer()
            try:
                # Validate message count
                if message_count <= 0:
                    await interaction.followup.send("*neural error detected!* Message count must be a positive number.")
                    return
                
                # Cap message count at 100
                message_count = min(message_count, 100)
                
                # Save the preference
                success = self.user_preferences_manager.set_channel_parsing(
                    str(interaction.user.id), 
                    enabled, 
                    message_count
                )
                
                if success:
                    status = "enabled" if enabled else "disabled"
                    await interaction.followup.send(f"*neural pathways reconfigured!* Channel message parsing is now **{status}**.\n\nWhen enabled, when you ping me I will include the last **{message_count}** messages from the channel in my context to better understand conversations and roleplay scenarios.")
                else:
                    await interaction.followup.send("*synaptic error detected!* Failed to save your preference. Please try again later.")
                    
            except Exception as e:
                logger.error(f"Error setting channel parsing preference: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while setting your preference.")
                
        @self.tree.command(name="query", description="Ask Publicia a question about Ledus Banum 77 and Imperial lore")
        @app_commands.describe(
            question="Your question about the lore",
            image_url="Optional URL to an image you want to analyze (must be a direct image URL ending with .jpg, .png, etc.)"
        )
        async def query_lore(interaction: discord.Interaction, question: str, image_url: str = None):
            try:
                # Try to defer immediately
                try:
                    await interaction.response.defer()
                except discord.errors.NotFound:
                    # If we get here, the interaction has expired
                    logger.warning("Interaction expired before we could defer")
                    return  # Exit gracefully
                except Exception as e:
                    logger.error(f"Error deferring interaction: {e}")
                    return  # Exit on any other error

                if not question:
                    await interaction.followup.send("*neural error detected!* Please provide a question.")
                    return

                # Get channel name and user info
                channel_name = interaction.channel.name if interaction.guild else "DM"
                nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name
                
                # Process image URL if provided
                image_attachments = []
                status_message = None
                
                if image_url:
                    try:
                        # Check if URL appears to be a direct image link
                        if any(image_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            status_message = await interaction.followup.send("*neural pathways activating... analyzing query and image...*", ephemeral=False)
                            
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
                                            await interaction.followup.send("*neural error detected!* The URL does not point to a valid image.", ephemeral=False)
                                            return
                                    else:
                                        await interaction.followup.send(f"*neural error detected!* Could not download image (status code: {resp.status}).", ephemeral=False)
                                        return
                        else:
                            await interaction.followup.send("*neural error detected!* The URL does not appear to be a direct image link. Please provide a URL ending with .jpg, .png, etc.", ephemeral=False)
                            return
                    except Exception as e:
                        logger.error(f"Error processing image URL: {e}")
                        await interaction.followup.send("*neural error detected!* Failed to process the image URL.", ephemeral=False)
                        return
                else:
                    status_message = await interaction.followup.send("*neural pathways activating... analyzing query...*", ephemeral=False)
                
                # Get user's preferred model
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )

                # Use the hybrid search system
                await status_message.edit(content="*analyzing query and searching imperial databases...*")
                search_results = self.process_hybrid_query(
                    question,
                    interaction.user.name,
                    max_results=self.config.get_top_k_for_model(preferred_model),
                    use_context=False  # this disables all the context logic for slash commands
                )
                
                # Log the results
                logger.info(f"Found {len(search_results)} relevant document sections")
                
                await status_message.edit(content="*synthesizing information...*")
                
                # Load Google Doc ID mapping for citation links
                googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

                # Extract image IDs from search results
                image_ids = []
                for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                    if image_id and image_id not in image_ids:
                        image_ids.append(image_id)
                        logger.info(f"Found relevant image: {image_id}")
                
                # Check if the question contains any Google Doc links
                doc_ids = await self._extract_google_doc_ids(question)
                google_doc_contents = []
                if doc_ids:
                    await status_message.edit(content="*detected Google Doc links in your query... fetching content...*")
                    for doc_id, doc_url in doc_ids:
                        content = await self._fetch_google_doc_content(doc_id)
                        if content:
                            google_doc_contents.append((doc_id, doc_url, content))

                # Format raw results with citation info
                import urllib.parse
                raw_doc_contexts = []
                for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                    if image_id:
                        # This is an image description
                        image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                        raw_doc_contexts.append(f"Image: {image_name} (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}")
                    elif doc in googledoc_mapping:
                        # Create citation link for Google Doc
                        doc_id = googledoc_mapping[doc]
                        words = chunk.split()
                        search_text = ' '.join(words[:min(10, len(words))])
                        encoded_search = urllib.parse.quote(search_text)
                        doc_url = f"https://docs.google.com/document/d/{doc_id}/"
                        raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
                    else:
                        raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) (similarity: {score:.2f}):\n{chunk}")

                # Add fetched Google Doc content to context
                google_doc_context = []
                for doc_id, doc_url, content in google_doc_contents:
                    # Truncate content if it's too long (first 2000 chars)
                    truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
                    google_doc_context.append(f"From Google Doc URL: {doc_url}:\n{truncated_content}")
                
                # Prepare messages for model
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]

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
                
                # Add image context if there are images from search or attachments
                if image_ids or image_attachments:
                    total_images = len(image_ids) + len(image_attachments)
                    img_source = []
                    if image_ids:
                        img_source.append(f"{len(image_ids)} from search results")
                    if image_attachments:
                        img_source.append(f"{len(image_attachments)} from attachments")
                    
                    messages.append({
                        "role": "system",
                        "content": f"The query has {total_images} relevant images ({', '.join(img_source)}). If you are a vision-capable model, you will see these images in the user's message."
                    })

                messages.append({
                    "role": "user",
                    "content": f"{nickname}: {question}"
                })

                # Get friendly model name based on the model value
                model_name = "Unknown Model"
                if "deepseek/deepseek-r1" in preferred_model:
                    model_name = "DeepSeek-R1"
                elif preferred_model.startswith("google/"):
                    model_name = "Gemini 2.0 Flash"
                elif preferred_model.startswith("nousresearch/"):
                    model_name = "Nous: Hermes 405B Instruct"
                elif "claude-3.5-haiku" in preferred_model:
                    model_name = "Claude 3.5 Haiku"
                elif "claude-3.5-sonnet" in preferred_model:
                    model_name = "Claude 3.5 Sonnet"
                elif "claude-3.7-sonnet" in preferred_model:
                    model_name = "Claude 3.7 Sonnet"
                elif "qwen/qwq-32b" in preferred_model:
                    model_name = "Qwen QwQ 32B"
                elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                    model_name = "Testing Model"
                elif "latitudegames/wayfarer" in preferred_model:
                    model_name = "Wayfarer 70B"
                elif "thedrummer/anubis-pro" in preferred_model:
                    model_name = "Anubis Pro 105B"

                if (image_attachments or image_ids) and preferred_model not in self.vision_capable_models:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*\n*note: your preferred model ({model_name}) doesn't support image analysis. only the text content will be processed.*")
                else:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")

                # Check if channel parsing is enabled for this user
                channel_parsing_enabled, channel_parsing_count = self.user_preferences_manager.get_channel_parsing(
                    str(interaction.user.id)
                )

                # If enabled, fetch channel messages and add to context
                if channel_parsing_enabled and channel_parsing_count > 0:
                    await status_message.edit(content="*analyzing channel conversation context...*")
                    
                    # Fetch channel messages
                    channel_messages = await self.fetch_channel_messages(
                        interaction.channel,
                        limit=channel_parsing_count,
                        max_message_length=500  # Limit each message to 500 characters
                    )
                    
                    if channel_messages:
                        # Format channel messages for context
                        channel_context = "Recent channel messages for context:\n\n"
                        for msg in channel_messages:
                            channel_context += f"{msg['author']}: {msg['content']}\n"
                        
                        # Limit total context size to 10000 characters
                        if len(channel_context) > 10000:
                            channel_context = channel_context[:10000] + "...\n(channel context truncated due to size)"
                            logger.warning(f"Channel context truncated to 10000 characters")
                        
                        # Add to messages array
                        messages.append({
                            "role": "system",
                            "content": channel_context
                        })
                        
                        logger.info(f"Added {len(channel_messages)} channel messages to context")
                        await status_message.edit(content=f"*analyzing query, search results, and channel context ({len(channel_messages)} messages)...*")
                
                # If enabled, fetch channel messages and add to context
                if channel_parsing_enabled and channel_parsing_count > 0:
                    await thinking_msg.edit(content="*analyzing channel conversation context...*")
                    
                    # Fetch channel messages
                    channel_messages = await self.fetch_channel_messages(
                        message.channel,
                        limit=channel_parsing_count,
                        max_message_length=500  # Limit each message to 500 characters
                    )
                    
                    if channel_messages:
                        # Format channel messages for context
                        channel_context = "Recent channel messages for context:\n\n"
                        for msg in channel_messages:
                            channel_context += f"{msg['author']}: {msg['content']}\n"
                        
                        # Limit total context size to 10000 characters
                        if len(channel_context) > 10000:
                            channel_context = channel_context[:10000] + "...\n(channel context truncated due to size)"
                            logger.warning(f"Channel context truncated to 10000 characters")
                        
                        # Add to messages array
                        messages.append({
                            "role": "system",
                            "content": channel_context
                        })
                        
                        logger.info(f"Added {len(channel_messages)} channel messages to context")
                        await thinking_msg.edit(content=f"*analyzing query, search results, and channel context ({len(channel_messages)} messages)...*")

                temperature = self.calculate_dynamic_temperature(
                    question
                )

                completion, actual_model = await self._try_ai_completion(
                    preferred_model,
                    messages,
                    image_ids=image_ids,
                    image_attachments=image_attachments,
                    temperature=temperature 
                )

                if completion and completion.get('choices') and len(completion['choices']) > 0:
                    if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                        response = completion['choices'][0]['message']['content']
                        
                        # Pass the status message as the existing_message parameter
                        await self.send_split_message(
                            interaction.channel,
                            response,
                            model_used=actual_model,
                            user_id=str(interaction.user.id),
                            existing_message=status_message
                        )
                    else:
                        logger.error(f"Unexpected response structure: {completion}")
                        await status_message.edit(content="*neural circuit overload!* I received an unexpected response structure.")
                else:
                    await interaction.followup.send("*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                try:
                    await interaction.followup.send("*neural circuit overload!* My brain is struggling and an error has occurred.")
                except:
                    logger.error("Failed to send error message to user")

        @self.tree.command(name="search_docs", description="Search the document knowledge base")
        @app_commands.describe(query="What to search for")
        async def search_documents(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                if not query:
                    await interaction.followup.send("*neural error detected!* Please provide a search query.")
                    return
                    
                results = self.document_manager.search(query, top_k=5)
                if not results:
                    await interaction.followup.send("No relevant documents found.")
                    return
                
                # Create batches of results that fit within Discord's message limit
                batches = []
                current_batch = "Search results:\n"
                
                for doc_name, chunk, similarity, image_id, chunk_index, total_chunks in results:
                    # Format this result
                    if image_id:
                        # This is an image search result
                        image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                        result_text = f"\n**IMAGE: {image_name}** (ID: {image_id}, similarity: {similarity:.2f}):\n"
                        result_text += f"```{self.sanitize_discord_text(chunk[:300])}...```\n"
                    else:
                        result_text = f"\n**From {doc_name}** (Chunk {chunk_index}/{total_chunks}) (similarity: {similarity:.2f}):\n"
                        result_text += f"```{self.sanitize_discord_text(chunk[:300])}...```\n"
                    
                    # Check if adding this result would exceed Discord's message limit
                    if len(current_batch) + len(result_text) > 1900:  # Leave room for Discord's limit
                        batches.append(current_batch)
                        current_batch = "Search results (continued):\n" + result_text
                    else:
                        current_batch += result_text
                
                # Add the last batch if it has content
                if current_batch and current_batch != "Search results (continued):\n":
                    batches.append(current_batch)
                
                # Send each batch as a separate message
                for batch in batches:
                    await interaction.followup.send(batch)
                    
            except Exception as e:
                await interaction.followup.send(f"Error searching documents: {str(e)}")

        @self.tree.command(name="remove_doc", description="Remove a document from the knowledge base")
        @app_commands.describe(name="Name of the document to remove")
        async def remove_document(interaction: discord.Interaction, name: str):
            await interaction.response.defer()
            try:
                if not name:
                    await interaction.followup.send("*neural error detected!* Please provide a document name.")
                    return
                    
                success = self.document_manager.delete_document(name)
                if success:
                    await interaction.followup.send(f"Removed document: {name} \n*google docs will also need to be removed from the tracked list*")
                else:
                    await interaction.followup.send(f"Document not found: {name}")
            except Exception as e:
                await interaction.followup.send(f"Error removing document: {str(e)}")

        @self.tree.command(name="list_files", description="List files that Publicia can upload")
        @app_commands.describe(
            file_type="Type of files to list",
            search_term="Optional search term to filter files"
        )
        @app_commands.choices(file_type=[
            app_commands.Choice(name="Documents", value="documents"),
            app_commands.Choice(name="Images", value="images"),
            app_commands.Choice(name="Lorebooks", value="lorebooks"),
            app_commands.Choice(name="All", value="all")
        ])
        async def list_files(interaction: discord.Interaction, file_type: str, search_term: str = None):
            await interaction.response.defer()
            try:
                files = []
                
                # Helper function to list files in a directory
                def get_files_from_dir(dir_path, prefix=""):
                    results = []
                    try:
                        for item in Path(dir_path).glob("*"):
                            if item.is_file() and not item.name.startswith(".") and not item.name.endswith(".pkl") and not item.name.endswith(".json"):
                                if search_term is None or search_term.lower() in item.name.lower():
                                    results.append(f"{prefix}{item.name}")
                    except Exception as e:
                        logger.error(f"Error listing files in {dir_path}: {e}")
                    return results
                
                # Get files based on type
                if file_type == "documents" or file_type == "all":
                    docs = get_files_from_dir(self.document_manager.base_dir, "[Document] ")
                    files.extend(docs)
                
                if file_type == "images" or file_type == "all":
                    # For images, get from metadata to include proper names
                    if self.image_manager.metadata:
                        for image_id, meta in self.image_manager.metadata.items():
                            name = meta.get('name', 'Unnamed')
                            if search_term is None or search_term.lower() in name.lower():
                                files.append(f"[Image] {name} (ID: {image_id})")
                
                if file_type == "lorebooks" or file_type == "all":
                    lorebooks = get_files_from_dir(self.document_manager.get_lorebooks_path(), "[Lorebook] ")
                    files.extend(lorebooks)
                
                # Send response
                if not files:
                    if search_term:
                        await interaction.followup.send(f"*neural search complete!* I couldn't find any files matching '{search_term}' in the selected storage areas.")
                    else:
                        await interaction.followup.send("*neural storage scan complete!* I don't have any files of this type in my memory banks.")
                    return
                
                # Create formatted response
                response = "*accessing neural storage banks...*\n\n**AVAILABLE FILES**\n\n"
                for i, file_name in enumerate(sorted(files)):
                    response += f"{i+1}. {file_name}\n"
                    
                # Add usage instructions
                response += "\n*To retrieve a file, use the `/retrieve_file` command with the file name (without type prefix)*"
                response += "\n*For images, use the image ID with the `/view_image` command*"
                
                # Split if necessary
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while trying to list files.")

        @self.tree.command(name="retrieve_file", description="Retrieve a file that Publicia has saved")
        @app_commands.describe(
            file_name="Name of the file to retrieve",
            file_type="Type of the file"
        )
        @app_commands.choices(file_type=[
            app_commands.Choice(name="Document", value="document"),
            app_commands.Choice(name="Lorebook", value="lorebook")
        ])
        async def retrieve_file(interaction: discord.Interaction, file_name: str, file_type: str):
            await interaction.response.defer()
            try:
                # Determine the directory based on file type
                if file_type == "document":
                    base_dir = self.document_manager.base_dir
                elif file_type == "lorebook":
                    base_dir = self.document_manager.get_lorebooks_path()
                else:
                    await interaction.followup.send("*neural error detected!* Invalid file type. Please select 'Document' or 'Lorebook'.")
                    return
                
                # Try exact match first
                path = Path(base_dir) / file_name
                
                # If not found, try adding .txt extension
                if not path.exists() and not file_name.endswith('.txt'):
                    path = Path(base_dir) / f"{file_name}.txt"
                
                # If still not found, try case-insensitive search
                if not path.exists():
                    found = False
                    try:
                        for item in Path(base_dir).glob("*"):
                            if item.is_file() and (item.name.lower() == file_name.lower() or 
                            item.name.lower() == f"{file_name.lower()}.txt"):
                                path = item
                                found = True
                                break
                    except Exception as e:
                        logger.error(f"Error searching for file: {e}")
                    
                    if not found:
                        await interaction.followup.send(f"*neural error detected!* Could not find file '{file_name}' in {file_type} storage.")
                        return
                
                # Check if the file exists and is a file
                if not path.exists() or not path.is_file():
                    await interaction.followup.send(f"*neural error detected!* Could not find file: {file_name}")
                    return
                
                # Send the file
                file = discord.File(path, filename=path.name)
                await interaction.followup.send(
                    content=f"*neural pathways accessing storage...* Here is the requested {file_type}: `{path.name}`",
                    file=file
                )
            
            except Exception as e:
                logger.error(f"Error retrieving file: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while trying to retrieve the file.")

        @self.tree.command(name="add_googledoc", description="Add a Google Doc to the tracked list")
        @app_commands.describe(
            doc_url="Google Doc URL or ID",
            name="Custom name for the document (optional)"
        )
        async def add_google_doc(interaction: discord.Interaction, doc_url: str, name: str = None):
            await interaction.response.defer()
            try:
                if not doc_url:
                    await interaction.followup.send("*neural error detected!* Please provide a Google Doc URL or ID.")
                    return
                    
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
                
            try:
                with open(tracked_file, 'r') as f:
                    tracked_docs = json.load(f)
                
                if not tracked_docs:
                    await interaction.followup.send("*my neural pathways show no connected google docs*")
                    return
                    
                response = "*accessing neural connections to google docs...*\n\n**TRACKED DOCUMENTS**\n"
                for doc in tracked_docs:
                    doc_id = doc['id']
                    name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                    doc_url = f"<https://docs.google.com/document/d/{doc_id}>"
                    response += f"\n{name} - URL: {doc_url}"
                
                # Split the message to avoid Discord's 2000 character limit
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            except Exception as e:
                logger.error(f"Error listing Google Docs: {e}")
                await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to list Google Docs.")

        @self.tree.command(name="rename_document", description="Rename any document, Google Doc, or lorebook")
        @app_commands.describe(
            current_name="Current name of the document to rename",
            new_name="New name for the document"
        )
        async def rename_document(interaction: discord.Interaction, current_name: str, new_name: str):
            await interaction.response.defer()
            try:
                if not current_name or not new_name:
                    await interaction.followup.send("*neural error detected!* Both current name and new name are required.")
                    return
                    
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
                if not identifier:
                    await interaction.followup.send("*neural error detected!* Please provide an identifier for the Google Doc to remove.")
                    return
                    
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

                local_file_name = doc_name
                if not local_file_name.endswith('.txt'):
                    local_file_name += '.txt'
                    
                # Remove local file if it exists
                local_file_path = Path(self.document_manager.base_dir) / local_file_name
                file_removed = False
                
                if local_file_path.exists():
                    try:
                        success = self.document_manager.delete_document(local_file_name)
                        if success:
                            file_removed = True
                        else:
                            await interaction.followup.send(f"Document tracked, but file not found in document manager: {local_file_name}")
                    except Exception as e:
                        await interaction.followup.send(f"Error removing document: {str(e)}")
                
                response = f"*I've surgically removed the neural connection to {doc_name}*\n*url: {doc_url}*"
                if file_removed:
                    response += f"\n*and removed the local document file ({local_file_name})*"
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
            except Exception as e:
                logger.error(f"Error removing Google Doc: {e}")
                await interaction.followup.send(f"*my enhanced brain experienced an error!* couldn't remove document: {str(e)}")

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

        @self.tree.command(name="reload_docs", description="Reload all documents from disk (admin only)")
        @app_commands.check(check_permissions)
        async def reload_docs(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                await self.document_manager.reload_documents()
                await interaction.followup.send("Documents reloaded successfully.")
            except Exception as e:
                await interaction.followup.send(f"Error reloading documents: {str(e)}")

        @self.tree.command(name="regenerate_embeddings", description="Regenerate all document embeddings (admin only)")
        @app_commands.check(check_permissions)
        async def regenerate_embeddings(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                status_message = await interaction.followup.send("*starting neural pathway recalibration...*")
                success = self.document_manager.regenerate_all_embeddings()
                if success:
                    await status_message.edit(content="*neural pathways successfully recalibrated!* All document embeddings have been regenerated.")
                else:
                    await status_message.edit(content="*neural pathway failure!* Failed to regenerate embeddings.")
            except Exception as e:
                logger.error(f"Error regenerating embeddings: {e}")
                await interaction.followup.send(f"*neural circuit overload!* Error regenerating embeddings: {str(e)}")

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

        @self.tree.command(name="help", description="Learn how to use Publicia and understand her capabilities and limitations")
        async def help_command(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                response = "# **PUBLICIA HELP GUIDE**\n\n"
                response += "*greetings, human! my genetically enhanced brain is ready to assist you with imperial knowledge. here's how to use my capabilities:*\n\n"
                
                # Core functionality
                response += "## **CORE FUNCTIONALITY**\n\n"
                response += "**🔍 Asking Questions**\n"
                response += "• **Mention me** in a message with your question about Ledus Banum 77 and Imperial lore\n"
                response += "• Use `/query` command for more structured questions (supports image URLs for analysis)\n"
                response += "• I'll search my knowledge base and provide answers with citations where possible\n"
                response += "• You can attach images directly to mentioned messages for visual analysis\n\n"
                response += "If you reply to a message and ping Publicia, she will be able to see the message you are replying to\n\n"
                
                # Knowledge Base
                response += "## **KNOWLEDGE BASE & LIMITATIONS**\n\n"
                response += "**📚 What I Know**\n"
                response += "• My knowledge is based on documents and images uploaded to my neural database\n"
                response += "• I specialize in Ledus Banum 77 (aka Tundra) lore and Imperial institutions\n"
                response += "• I can cite specific documents when providing information\n"
                response += "• I understand the Infinite Empire's structure, planes of existence, and Resonant Drilling\n\n"
                
                response += "**⚠️ What I Don't Know**\n"
                response += "• Information not contained in my document or image database\n"
                response += "• I cannot access the internet\n"
                response += "• I am bad at highly broad queries, or at queries asking for info that is not in my knowledge base\n"
                response += "• I am bad at queries that would require information from many different sources, as my embedding search system has a limit on the amount of document chunks it will return\n"
                response += "   • I would recommend breaking down your query into smaller, more focused questions so that my embeddings search can return more relevant and focused results\n"
                response += "• I may lack up to date information if my documents have not been updated\n\n"
                
                # How I Work
                response += "## **HOW I WORK**\n\n"
                response += "**🧠 Neural Processing**\n"
                response += "• I use semantic search with advanced reranking to find relevant information\n"
                response += "• I analyze your query to understand what you're looking for\n"
                response += "• I synthesize information from multiple documents when needed\n"
                response += "• I provide citations to document sources when possible\n"
                response += "• I automatically extract content from Google Docs linked in your queries\n"
                response += "• I use dynamic temperature control to adapt my responses to your query type\n\n"
                
                response += "**🖼️ Image Analysis**\n"
                response += "• I can analyze images in three ways:\n"
                response += "  - Attach an image directly when mentioning me\n"
                response += "  - Use `/query` with an image URL\n"
                response += "  - I can search my image database for relevant visual information\n"
                response += "• I can recognize content in images and integrate them into my responses\n"
                response += "• Add images to my knowledge base using `Publicia! add_image` for future searches\n"
                response += "• Vision-capable models: Gemini 2.0 Flash, Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet\n\n"
                
                # Document Management
                response += "## **DOCUMENT & IMAGE MANAGEMENT**\n\n"
                response += "**📚 Adding Information**\n"
                response += "• `/add_info` - Add text directly to my knowledge base\n"
                response += "• `Publicia! add_doc` - Add a document with an attachment\n"
                response += "• `/add_googledoc` - Connect a Google Doc to my knowledge base\n"
                response += "• `Publicia! add_image \"name\" [yes/no]` - Add an image with optional auto-description\n\n"
                
                response += "**📋 Managing Documents & Images**\n"
                response += "• `/list_docs` - See all documents in my knowledge base\n"
                response += "• `/list_images` - See all images in my visual knowledge base\n"
                response += "• `/list_files` - See all files in my knowledge base\n"
                response += "• `/retrieve_file` - Retrieve a file from my knowledge base and upload it to Discord\n\n"
                response += "• `/view_image` - View an image from my knowledge base\n"
                response += "• `/remove_doc` - Remove a document from my knowledge base\n"
                response += "• `/remove_image` - Remove an image from my knowledge base\n"
                response += "• `/remove_googledoc` - Disconnect a Google Doc\n"
                response += "• `/rename_document` - Rename a document in my database\n"
                response += "• `/search_docs` - Search directly in my document knowledge base\n"
                response += "• `/update_image_description` - Update the description for an image\n"
                response += "• `/reload_docs` - Reload all documents from disk (admin only)\n\n"
                
                # Conversation Management
                response += "## **CONVERSATION SYSTEM**\n\n"
                response += "**💬 How Conversations Work**\n"
                response += "• I remember your chats to provide more relevant, contextual responses\n"
                response += "• Each user has their own conversation history stored separately\n"
                response += "• When you ask something, I check your previous interactions for context\n"
                response += "• This lets me understand ongoing discussions, recurring topics, and your interests\n"
                response += "• Conversations are limited to the most recent 50 messages to maintain performance\n"
                response += "• Use `/parse_channel` to let me analyze recent channel messages for more context\n\n"

                response += "**🧠 Memory Management**\n"
                response += "• `/history [limit]` - See your recent conversation (default: shows last 10 messages)\n"
                response += "• `/manage_history [limit]` - View messages with numbered indices for selective deletion\n"
                response += "• `/delete_history_messages indices:\"0,2,5\" confirm:\"yes\"` - Remove specific messages by their indices\n"
                response += "• `/lobotomise` command to completely wipe your history\n"
                response += "• `/archive_conversation [archive_name]` - Save your current conversation history with optional custom name\n" 
                response += "• `/list_archives` - View all your archived conversations\n"
                response += "• `/swap_conversation archive_name` - Switch between current and archived conversations (automatically saves current conversation first)\n"
                response += "• `/delete_archive archive_name confirm:\"yes\"` - Permanently delete an archived conversation (requires confirmation)\n"
                response += "• Memory management lets you organize conversations, preserve important discussions, and start fresh when needed\n\n"
                
                # Customization
                response += "## **CUSTOMIZATION**\n\n"
                response += "**⚙️ AI Model Selection**\n"
                response += "• `/set_model` - Choose your preferred AI model:\n"
                response += "• `/get_model` - Check which model you're currently using, as well as get a list of all available models and their descriptions\n"
                response += "• `/toggle_debug` - Show/hide which model generated each response\n\n"
                response += "I recommend using Gemini 2.0 Flash for factual queries, DeepSeek-R1 for times when you want good prose and creative writing, Claude 3.5 Haiku for roleplay and accuracy, and QwQ 32B when you want a balance.\n\n"
                
                # Add our new section here
                response += "**🧪 Debugging Tools**\n"
                response += "• `/export_prompt` - Export the complete prompt for your query\n"
                response += "  - Shows system prompt, conversation history, search results, and more\n"
                response += "  - Helps understand exactly how I process your questions\n"
                response += "  - Includes privacy option to make output only visible to you\n\n"
                
                # Tips
                response += "## **TIPS FOR BEST RESULTS**\n\n"
                response += "• Ask specific questions for more accurate answers\n"
                response += "• If I don't know something, add relevant documents or images to my database\n"
                response += "• Use Google Docs integration for large, regularly updated documents\n"
                response += "• Include links to Google Docs in your queries for on-the-fly context\n"
                response += "• For creative writing, try DeepSeek-R1\n"
                response += "• For factual accuracy, try Gemini 2.0 Flash or Claude 3.5 Haiku\n\n"
                response += "*my genetically enhanced brain is always ready to help... just ask!*"
                
                # Send the response in chunks
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
            except Exception as e:
                logger.error(f"Error displaying help: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while trying to display help information.")
                
                # Send the response in chunks
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
                    
            except Exception as e:
                logger.error(f"Error displaying help: {e}")
                await interaction.followup.send("*neural circuit overload!* An error occurred while trying to display help information.")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        
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
    
    async def _try_ai_completion(self, model: str, messages: List[Dict], image_ids=None, image_attachments=None, temperature=0.1, max_retries=2, min_response_length=5, **kwargs) -> Tuple[Optional[Any], Optional[str]]:
        """Get AI completion with dynamic fallback options based on the requested model.
        
        Returns:
            Tuple[Optional[Dict], Optional[str]]: (completion result, actual model used)
        """
        # Get primary model family (deepseek, google, etc.)
        model_family = model.split('/')[0] if '/' in model else None
        
        # Check if we need a vision-capable model
        need_vision = (image_ids and len(image_ids) > 0) or (image_attachments and len(image_attachments) > 0)
        
        # Build fallback list dynamically based on the requested model
        models = [model]  # Start with the requested model
        
        # Add model-specific fallbacks first
        if model_family == "deepseek":
            fallbacks = [
                "deepseek/deepseek-r1:free",
                #"deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1:floor",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-r1:nitro",
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-distill-qwen-32b"
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "qwen":  # Add Qwen fallbacks
            fallbacks = [
                "qwen/qwq-32b:free",
                "qwen/qwq-32b",
                "qwen/qwen-turbo",
                "qwen/qwen2.5-32b-instruct"
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "google":
            fallbacks = [
                "google/gemini-2.0-flash-thinking-exp:free",
                "google/gemini-2.0-pro-exp-02-05:free",
                "google/gemini-2.0-flash-001"
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "thedrummer":  # Testing Model fallbacks
            fallbacks = [
                "thedrummer/unslopnemo-12b",
                "thedrummer/rocinante-12b",
                "meta-llama/llama-3.3-70b-instruct"  # Safe fallback option
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "eva-unit-01":
            fallbacks = [
                "eva-unit-01/eva-qwen-2.5-72b:floor",
                "eva-unit-01/eva-qwen-2.5-72b",
                "qwen/qwq-32b:free",
                "qwen/qwq-32b",
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "nousresearch":
            fallbacks = [
                "nousresearch/hermes-3-llama-3.1-70b",
                "meta-llama/llama-3.3-70b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct"
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "anthropic":
            if "claude-3.7-sonnet" in model:
                fallbacks = [
                    "anthropic/claude-3.7-sonnet",
                    "anthropic/claude-3.5-sonnet:beta",
                    "anthropic/claude-3.5-haiku:beta",
                    "anthropic/claude-3.5-haiku"
                ]
            elif "claude-3.5-sonnet" in model:
                fallbacks = [
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3.7-sonnet:beta",
                    "anthropic/claude-3.7-sonnet",
                    "anthropic/claude-3.5-haiku:beta",
                    "anthropic/claude-3.5-haiku"
                ]
            elif "claude-3.5-haiku" in model:
                fallbacks = [
                    "anthropic/claude-3.5-haiku",
                    "anthropic/claude-3-haiku:beta"
                ]
            else:
                fallbacks = []
            models.extend([fb for fb in fallbacks if fb not in models])
        # Inside the _try_ai_completion method
        elif model_family == "latitudegames":
            fallbacks = [
                "latitudegames/wayfarer-large-70b-llama-3.3",
                "meta-llama/llama-3.3-70b-instruct",  # base model fallback
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "thedrummer" and "anubis" in model:
            fallbacks = [
                "thedrummer/anubis-pro-105b-v1",
                "latitudegames/wayfarer-large-70b-llama-3.3",  # try other narrative model
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        
        # Add general fallbacks that aren't already in the list
        general_fallbacks = [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-r1:free",
            "qwen/qwq-32b:free",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
            "google/gemini-2.0-pro-exp-02-05:free",
            "nousresearch/hermes-3-llama-3.1-405b",
            "anthropic/claude-3.5-haiku:beta",
            "anthropic/claude-3.5-haiku"
        ]
        models.extend([fb for fb in general_fallbacks if fb not in models])

        # Headers for API calls
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "Publicia - DPS",
            "Content-Type": "application/json"
        }

        for current_model in models:
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
                        if model != current_model:
                            logger.info(f"Notice: Fallback model {current_model} was used instead of requested {model}")
                            
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
        
        logger.error(f"All models failed to generate completion. Attempted models: {', '.join(models)}")
        return None, None  # Return None for both completion and model used

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

            referenced_message = None
            if message.reference and message.reference.resolved:
                referenced_message = message.reference.resolved
                logger.info(f"Message is a reply to a message from {referenced_message.author.name}: {shorten(referenced_message.content, width=100, placeholder='...')}")
                            
            
            logger.info(f"Processing message from {message.author.name}: {shorten(message.content, width=100, placeholder='...')}")

            # Extract the question from the message (remove mentions)
            question = message.content
            for mention in message.mentions:
                question = question.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
            question = question.strip()
            
            # Check if the stripped question is empty
            if not question:
                question = "Hello"
                logger.info("Received empty message after stripping mentions, defaulting to 'Hello'")
            
            # Add context-aware query enhancement
            original_question = question
            # Check if this query might need context
            if self.is_context_dependent_query(question):
                # Get context from conversation history
                context = self.get_conversation_context(message.author.name, question)
                
                if context:
                    # Enhance the query with context
                    question = self.enhance_context_dependent_query(question, context)
                    logger.info(f"Enhanced query: '{original_question}' -> '{question}'")
            
            # Check for Google Doc links in the message
            google_doc_ids = await self._extract_google_doc_ids(question)
            google_doc_contents = []
            
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
                    if is_image(attachment):
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
            
            # Get user's preferred model
            preferred_model = self.user_preferences_manager.get_preferred_model(
                str(message.author.id),
                default_model=self.config.LLM_MODEL
            )

            # Update thinking message
            await thinking_msg.edit(content="*analyzing query and searching imperial databases...*")

            # Use the new hybrid search system
            search_results = self.process_hybrid_query(
                question,
                message.author.name,
                max_results=self.config.get_top_k_for_model(preferred_model)
            )
            
            # Extract results
            synthesis = ""  # No synthesis in hybrid search
            
            # Log the results
            logger.info(f"Found {len(search_results)} relevant document sections")

            # Load Google Doc ID mapping for citation links
            googledoc_mapping = self.document_manager.get_googledoc_id_mapping()

            # Extract image IDs from search results
            image_ids = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
                    logger.info(f"Found relevant image: {image_id}")

            # Fetch content for Google Doc links
            if google_doc_ids:
                # Fetch content for each Google Doc
                await thinking_msg.edit(content="*detected Google Doc links in your query... fetching content...*")
                for doc_id, doc_url in google_doc_ids:
                    content = await self._fetch_google_doc_content(doc_id)
                    if content:
                        logger.info(f"Fetched content from Google Doc {doc_id}")
                        google_doc_contents.append((doc_id, doc_url, content))

            # Format raw results with citation info
            import urllib.parse
            raw_doc_contexts = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id:
                    # This is an image description
                    image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                    raw_doc_contexts.append(f"Image: {image_name} (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}")
                elif doc in googledoc_mapping:
                    # Create citation link for Google Doc
                    doc_id = googledoc_mapping[doc]
                    words = chunk.split()
                    search_text = ' '.join(words[:min(10, len(words))])
                    encoded_search = urllib.parse.quote(search_text)
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/"
                    raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
                else:
                    raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) (similarity: {score:.2f}):\n{chunk}")

            # Add fetched Google Doc content to context
            google_doc_context = []
            for doc_id, doc_url, content in google_doc_contents:
                # Truncate content if it's too long (first 10000 chars)
                truncated_content = content[:10000] + ("..." if len(content) > 10000 else "")
                google_doc_context.append(f"From Google Doc URL: {doc_url}:\n{truncated_content}")
            
            # Get nickname or username
            nickname = None
            if message.guild and hasattr(message.author, 'nick') and message.author.nick:
                nickname = message.author.nick
            else:
                nickname = message.author.name

            # Prepare messages for model
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *conversation_messages
            ]

            if referenced_message:
                # Get the author name/nickname
                reply_author = None
                if message.guild and hasattr(referenced_message.author, 'nick') and referenced_message.author.nick:
                    reply_author = referenced_message.author.nick
                else:
                    reply_author = referenced_message.author.name                
                # Process the content to handle mentions
                ref_content = referenced_message.content
                for mention in referenced_message.mentions:
                    mention_name = None
                    if message.guild and hasattr(mention, 'nick') and mention.nick:
                        mention_name = mention.nick
                    else:
                        mention_name = mention.name
                    ref_content = ref_content.replace(f'<@{mention.id}>', f'@{mention_name}').replace(f'<@!{mention.id}>', f'@{mention_name}')
                
                # Check for attachments
                attachment_info = ""
                if referenced_message.attachments:
                    attachment_count = len(referenced_message.attachments)
                    attachment_info = f" [with {attachment_count} attachment{'s' if attachment_count > 1 else ''}]"
                
                # Check if it's a message from Publicia herself
                if referenced_message.author == self.user:
                    messages.append({
                        "role": "system",
                        "content": f"The user is replying to your previous message: \"{ref_content}\"{attachment_info}"
                    })
                else:
                    messages.append({
                        "role": "system",
                        "content": f"The user is replying to a message from {reply_author}: \"{ref_content}\"{attachment_info}"
                    })

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
                "role": "system",
                "content": f"You are responding to a message in the Discord channel: {channel_name}"
            })
            
            # Add image context if there are images from search or attachments
            if image_ids or image_attachments:
                total_images = len(image_ids) + len(image_attachments)
                img_source = []
                if image_ids:
                    img_source.append(f"{len(image_ids)} from search results")
                if image_attachments:
                    img_source.append(f"{len(image_attachments)} from attachments")
                    
                messages.append({
                    "role": "system",
                    "content": f"The query has {total_images} relevant images ({', '.join(img_source)}). If you are a vision-capable model, you will see these images in the user's message."
                })

            """messages.append({
                "role": "system",
                "content": "IMPORTANT: Do not make up or be incorrect about information about the setting of Ledus Banum 77 or the Infinite Empire. If you don't have information on what the user is asking, admit that you don't know."
            })"""
            
            messages.append({
                "role": "user",
                "content": f"{nickname}: {original_question}"
            })

            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            if "deepseek/deepseek-r1" in preferred_model:
                model_name = "DeepSeek-R1"
            elif preferred_model.startswith("google/"):
                model_name = "Gemini 2.0 Flash"
            elif preferred_model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in preferred_model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-3.5-sonnet" in preferred_model:
                model_name = "Claude 3.5 Sonnet"
            elif "claude-3.7-sonnet" in preferred_model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in preferred_model:
                model_name = "Qwen QwQ 32B"
            elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in preferred_model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in preferred_model:
                model_name = "Anubis Pro 105B"

            # Add a note about vision capabilities if relevant
            if (image_attachments or image_ids) and preferred_model not in self.vision_capable_models:
                await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(note: your preferred model ({model_name}) doesn't support image analysis. only the text content will be processed)*")
                # No model switching - continues with user's preferred model
            else:
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(fetched content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})*")
                else:
                    await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")

            # Calculate dynamic temperature based on query and conversation history
            temperature = self.calculate_dynamic_temperature(
                question,
                conversation_messages
            )
            
            # Get AI response using user's preferred model
            completion, actual_model = await self._try_ai_completion(
                preferred_model,
                messages,
                image_ids=image_ids,
                image_attachments=image_attachments,
                temperature=temperature
            )

            if completion and completion.get('choices') and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                    response = completion['choices'][0]['message']['content']
                else:
                    logger.error(f"Unexpected response structure: {completion}")
                    await thinking_msg.edit(content="*neural circuit overload!* I received an unexpected response structure.")
                    return
                
                # Update conversation history
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "user",
                    original_question + (" [with image attachment(s)]" if image_attachments else ""),
                    channel_name
                )
                self.conversation_manager.write_conversation(
                    message.author.name,
                    "assistant",
                    response,
                    channel_name
                )

                # Send the response, replacing thinking message with the first chunk
                await self.send_split_message(
                    message.channel,
                    response,
                    reference=message,
                    mention_author=False,
                    model_used=actual_model,  # Pass the actual model used, not just the preferred model
                    user_id=str(message.author.id),
                    existing_message=thinking_msg
                )
            else:
                await thinking_msg.edit(content="*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            try:
                await message.channel.send(
                    "*neural circuit overload!* My brain is struggling and an error has occurred.",
                    reference=message,
                    mention_author=False
                )
            except:
                pass  # If even sending the error message fails, just log and move on

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

    def enhance_context_dependent_query(self, query: str, context: str) -> str:
        """Enhance a context-dependent query with conversation context."""
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

    def generate_context_aware_embedding(self, query: str, context: str):
        """Generate an embedding that combines current query with context."""
        if not context:
            # No context, use normal embedding
            return self.document_manager.generate_embeddings([query], is_query=True)[0]
        
        # Generate embeddings for different query variants
        query_variants = [
            query,                   # Original query (highest weight)
            f"{query} {context}",    # Query + context
            context                  # Just context (lowest weight)
        ]
        
        embeddings = self.document_manager.generate_embeddings(query_variants, is_query=True)
        
        # Weight: 60% original query, 30% combined, 10% context
        weighted_embedding = 0.6 * embeddings[0] + 0.3 * embeddings[1] + 0.1 * embeddings[2]
        
        # Normalize the embedding
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm
            
        return weighted_embedding

    def process_hybrid_query(self, question: str, username: str, max_results: int = 5, use_context: bool = True):
        """Process queries using a hybrid of caching and context-aware embeddings with re-ranking."""
        # Skip context logic completely if use_context is False
        if not use_context:
            # Just do regular search with reranking
            search_results = self.document_manager.search(
                question, 
                top_k=max_results,
                apply_reranking=self.config.RERANKING_ENABLED
            )
            
            # Still cache results for consistency
            self.cache_search_results(username, question, search_results)
            return search_results
    
        is_followup = self.is_context_dependent_query(question)
        original_question = question
        
        # For standard non-follow-up queries
        if not is_followup:
            # Determine whether to apply re-ranking
            apply_reranking = self.config.RERANKING_ENABLED
            
            # Do regular search with re-ranking
            search_results = self.document_manager.search(
                question, 
                top_k=max_results,
                apply_reranking=apply_reranking
            )
            
            # Cache for future follow-ups
            self.cache_search_results(username, question, search_results)
            return search_results
        
        # For follow-up queries
        logger.info(f"Detected follow-up query: '{question}'")
        
        # Try to get more results from previous search
        cached_results = self.get_additional_results(username, top_k=max_results)
        
        if cached_results:
            # We have unused results, no need for new search
            logger.info(f"Using {len(cached_results)} cached results")
            return cached_results
        
        # No cached results, use context-aware search
        logger.info("No cached results, performing context-aware search")
        
        # Get conversation context
        context = self.get_conversation_context(username, question)
        
        if context:
            logger.info(f"Using context from conversation: '{context}'")
            
            # Generate context-aware embedding
            embedding = self.generate_context_aware_embedding(question, context)
            
            # Search with this embedding and apply re-ranking
            apply_reranking = self.config.RERANKING_ENABLED
            
            if apply_reranking:
                # Get more initial results for re-ranking
                initial_results = self.document_manager.custom_search_with_embedding(
                    embedding, 
                    top_k=self.config.RERANKING_CANDIDATES
                )
                
                # Apply re-ranking
                if initial_results:
                    logger.info(f"Applying re-ranking to {len(initial_results)} context-aware results")
                    results = self.document_manager.rerank_results(question, initial_results, top_k=max_results)
                    return results
            
            # If re-ranking is disabled or failed, use standard search
            results = self.document_manager.custom_search_with_embedding(embedding, top_k=max_results)
            return results
        else:
            # Fallback to normal search with re-ranking
            logger.info("No context found, using standard search with re-ranking")
            search_results = self.document_manager.search(
                question, 
                top_k=max_results,
                apply_reranking=self.config.RERANKING_ENABLED
            )
            return search_results

async def main():
    try:
        display_startup_banner()
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

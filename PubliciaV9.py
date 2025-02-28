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


# Reconfigure stdout to use UTF-8 with error replacement
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def is_image(attachment):
    """Check if an attachment is an image based on content type or file extension."""
    if attachment.content_type and attachment.content_type.startswith('image/'):
        return True
    # Fallback to checking file extension
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
    return any(attachment.filename.lower().endswith(ext) for ext in image_extensions)


def split_message(text, max_length=1750):
    """
    Split a text string into chunks under max_length characters,
    preserving newlines where possible but avoiding unnecessary splits.
    """
    if not text:
        return []
        
    # If the text is shorter than max_length, return it as a single chunk
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first (double newlines)
    paragraphs = text.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        # Check if adding this paragraph (with double newline if not the first paragraph)
        # would exceed the max length
        separator = '\n\n' if current_chunk and i > 0 else ''
        if len(current_chunk + separator + paragraph) <= max_length:
            # We can add this paragraph to the current chunk
            current_chunk = current_chunk + separator + paragraph
        else:
            # Check if the current chunk has content
            if current_chunk:
                chunks.append(current_chunk)
            
            # If the paragraph itself is longer than max_length, split it by lines
            if len(paragraph) > max_length:
                lines = paragraph.split('\n')
                current_chunk = ""
                
                for line in lines:
                    if len(current_chunk + ('' if not current_chunk else '\n') + line) <= max_length:
                        current_chunk = current_chunk + ('' if not current_chunk else '\n') + line
                    else:
                        # If the line itself is too long
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        
                        # Split the line into smaller chunks if needed
                        if len(line) > max_length:
                            for j in range(0, len(line), max_length):
                                line_chunk = line[j:j + max_length]
                                if j + max_length >= len(line):
                                    current_chunk = line_chunk
                                else:
                                    chunks.append(line_chunk)
                        else:
                            current_chunk = line
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # Final safety check: ensure no chunk exceeds max_length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Split any chunk that somehow exceeds max_length
            for j in range(0, len(chunk), max_length):
                final_chunks.append(chunk[j:j + max_length])
    
    return final_chunks
        
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
        # Handle empty text
        if not text or not text.strip():
            return []
            
        words = text.split()
        
        # Handle text with too few words
        if not words:
            return []
        
        if len(words) <= chunk_size:
            return [' '.join(words)]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            # Ensure we don't go out of bounds
            end_idx = min(i + chunk_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
            
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
    
    def add_document(self, name: str, content: str, save_to_disk: bool = True):
        """Add a new document to the system."""
        try:
            # Check if content is empty
            if not content or not content.strip():
                logger.warning(f"Document {name} has no content. Skipping.")
                return
                
            # Create chunks
            chunks = self._chunk_text(content)
            
            # Check if we have any chunks
            if not chunks:
                logger.warning(f"Document {name} has no content to chunk. Skipping.")
                return
            
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
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, str, float, Optional[str]]]:
        """
        Search for relevant document chunks.
        Returns a list of tuples (doc_name, chunk, similarity_score, image_id_if_applicable)
        """
        try:
            # Use instance top_k if none provided
            if top_k is None:
                top_k = self.top_k
            
            # Check if query is empty
            if not query or not query.strip():
                logger.warning("Empty query provided to search")
                return []
                
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            results = []
            logger.info(f"Searching documents for query: {shorten(query, width=100, placeholder='...')}")
            
            # Search each document
            for doc_name, doc_embeddings in self.embeddings.items():
                # Skip empty embeddings
                if len(doc_embeddings) == 0:
                    logger.warning(f"Skipping document {doc_name} with empty embeddings")
                    continue
                    
                # Calculate similarities
                try:
                    similarities = np.dot(doc_embeddings, query_embedding) / (
                        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
                    )
                    
                    # Get top chunks - make sure to handle edge cases
                    if len(similarities) > 0:
                        top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):]
                        
                        for idx in top_indices:
                            # Check if this is an image description
                            image_id = None
                            if doc_name.startswith("image_") and doc_name.endswith(".txt"):
                                # Extract image ID from document name
                                image_id = doc_name[6:-4]  # Remove "image_" prefix and ".txt" suffix
                            
                            # Check if metadata has image_id set explicitly
                            elif doc_name in self.metadata and 'image_id' in self.metadata[doc_name]:
                                image_id = self.metadata[doc_name]['image_id']
                            
                            # Make sure chunk index is valid
                            if idx < len(self.chunks[doc_name]):
                                results.append((
                                    doc_name,
                                    self.chunks[doc_name][idx],
                                    float(similarities[idx]),
                                    image_id
                                ))
                except Exception as e:
                    logger.error(f"Error calculating similarities for document {doc_name}: {e}")
                    continue
            
            # Sort by similarity
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Log search results
            for doc_name, chunk, similarity, image_id in results[:top_k]:
                logger.info(f"Found relevant chunk in {doc_name} (similarity: {similarity:.2f})")
                if image_id:
                    logger.info(f"This is an image description for image ID: {image_id}")
                logger.info(f"Chunk content: {shorten(sanitize_for_logging(chunk), width=300, placeholder='...')}")
            
            return results[:top_k]
            
        except KeyError as e:
            logger.error(f"Document or chunk not found in search: {e}")
            return []
        except ValueError as e:
            logger.error(f"Value error in document search: {e}")
            return []
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

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        
        # Configure models with defaults
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'google/gemini-2.0-flash-001')  # Default to Gemini
        self.CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'google/gemini-2.0-flash-001')  # Default to Gemini
        
        self.TOP_K = int(os.getenv('TOP_K', '10'))
        
        # Validate required environment variables
        self._validate_config()
        
        # Add timeout settings
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', '180'))
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

        # Clean up any empty documents - THIS IS THE NEW CODE
        empty_docs = self.document_manager.cleanup_empty_documents()
        if empty_docs:
            logger.info(f"Cleaned up {len(empty_docs)} empty documents at startup")
        
        # Initialize the ImageManager with a reference to the DocumentManager
        self.image_manager = ImageManager(document_manager=self.document_manager)
        
        self.user_preferences_manager = UserPreferencesManager()
        
        self.timeout_duration = 150

        self.banned_users = set()
        self.banned_users_file = 'banned_users.json'
        self.load_banned_users()

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.refresh_google_docs, 'interval', hours=6)
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
        """Use the configured classifier model to analyze the query and extract keywords/topics."""
        try:
            # Check if query is empty
            if not query or not query.strip():
                logger.warning("Empty query provided to analyze_query")
                return {"success": False}
                
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

    async def enhanced_search(self, query: str, analysis: Dict) -> List[Tuple[str, str, float, Optional[str]]]:
        """Perform an enhanced search based on query analysis."""
        try:
            # Check if query is empty
            if not query or not query.strip():
                logger.warning("Empty query provided to enhanced_search")
                return []
                
            # Check if analysis is a dictionary and has the success key
            if not isinstance(analysis, dict) or not analysis.get("success", False):
                # Fall back to basic search if analysis failed or is not a dict
                logger.info("Analysis is not valid, falling back to basic search")
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
            
            # Log found image results
            image_count = sum(1 for _, _, _, img_id in search_results if img_id)
            if image_count > 0:
                logger.info(f"Search found {image_count} relevant image results")
            
            return search_results
                
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return self.document_manager.search(query)  # Fall back to basic search

    async def synthesize_results(self, query: str, search_results: List[Tuple[str, str, float, Optional[str]]], analysis: Dict) -> str:
        """Use the configured classifier model to synthesize search results into a coherent context."""
        try:
            # Check if query is empty
            if not query or not query.strip():
                logger.warning("Empty query provided to synthesize_results")
                return ""
                
            # Format search results into a string, handling image descriptions
            result_text = ""
            for doc, chunk, score, image_id in search_results[:10]:  # Limit to top 10 results
                if image_id:
                    # This is an image description
                    try:
                        image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                        result_text += f"\nImage: {image_name} (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}\n"
                    except KeyError:
                        # Handle missing image metadata
                        result_text += f"\nImage: Unknown (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}\n"
                else:
                    # Regular document
                    result_text += f"\nDocument: {doc}\nContent: {chunk}\nRelevance: {score:.2f}\n"
            
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
                    6. Identify any images that appear in the search results and incorporate them in your synthesis
                    
                    Synthesize this information into a coherent context that can be used to answer the query.
                    Focus on extracting and organizing the facts, not on answering the query directly.
                    Include any citation information found in the document sections.
                    If there are relevant images, indicate where they should be referenced in responses.
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

    async def send_split_message(self, channel, text, reference=None, mention_author=False, model_used=None, user_id=None, existing_message=None):
        """Send a message split into chunks if it's too long, with each chunk referencing the previous one."""
        chunks = split_message(text)
        
        if model_used and user_id:
            debug_mode = self.user_preferences_manager.get_debug_mode(user_id)
            if debug_mode:
                # Format the debug info to show the actual model used
                # Keep the full model identifier to be transparent about which exact model was used
                debug_info = f"\n\n*[Debug: Response generated using {model_used}]*"
                
                # Check if adding debug info would exceed the character limit
                if len(chunks[-1]) + len(debug_info) > 1750:
                    # Create a new chunk for the debug info
                    chunks.append(debug_info)
                else:
                    chunks[-1] += debug_info
        
        # Keep track of the last message sent to use as reference for the next chunk
        last_message = None
        
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
            try:
                first_message = await channel.send(
                    content=chunks[0],
                    reference=reference,
                    mention_author=mention_author
                )
                last_message = first_message
                chunks = chunks[1:]  # Remove the first chunk since it's already sent
            except Exception as e:
                logger.error(f"Error sending first message chunk: {e}")
                return
        
        # Send remaining chunks sequentially, each referencing the previous one
        for chunk in chunks:
            try:
                # Each new chunk references the previous one to maintain the chain
                new_message = await channel.send(
                    content=chunk,
                    reference=last_message,  # Reference the previous message in the chain
                    mention_author=False  # Don't mention for follow-up chunks
                )
                # Update reference for the next chunk
                last_message = new_message
            except Exception as e:
                logger.error(f"Error sending message chunk: {e}")
                # Continue with the next chunk anyway

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
        except Exception as e:
            logger.error(f"Error refreshing Google Docs: {e}")
            
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
                    "Document Management": ["add_info", "list_docs", "remove_doc", "search_docs", "add_googledoc", "list_googledocs", "remove_googledoc", "rename_document"],
                    "Image Management": ["list_images", "view_image", "remove_image", "update_image_description"],
                    "Utility": ["list_commands", "set_model", "get_model", "toggle_debug", "help"],
                    "Memory Management": ["lobotomise", "history", "manage_history", "delete_history_messages"], 
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
                    "Utility": []
                }
                
                # Sort commands into categories
                for cmd in prefix_commands:
                    if "doc" in cmd.name.lower():
                        prefix_categories["Document Management"].append(cmd)
                    elif "image" in cmd.name.lower():
                        prefix_categories["Image Management"].append(cmd)
                    else:
                        prefix_categories["Utility"].append(cmd)
                
                # Format and add each category of prefix commands
                for category, cmds in prefix_categories.items():
                    if cmds:  # Only show categories that have commands
                        response += f"__*{category}*__\n"
                        for cmd in cmds:
                            brief = cmd.brief or "No description available"
                            response += f"`Publicia! {cmd.name}`: {brief}\n"
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
            app_commands.Choice(name="DeepSeek-R1 (best for immersive roleplaying and creative responses, but is slowe to respond)", value="deepseek/deepseek-r1:free"),
            app_commands.Choice(name="Gemini 2.0 Flash (best for accuracy, citations, and image analysis, and is very fast)", value="google/gemini-2.0-flash-001"),
            app_commands.Choice(name="Nous: Hermes 405B (balanced between creativity and factual precision)", value="nousresearch/hermes-3-llama-3.1-405b"),
            app_commands.Choice(name="Claude 3.5 Haiku (fast responses with image capabilities)", value="anthropic/claude-3.5-haiku:beta"),
            app_commands.Choice(name="Claude 3.5 Sonnet (admin only, premium all-around capabilities)", value="anthropic/claude-3.5-sonnet:beta"),
            app_commands.Choice(name="Claude 3.7 Sonnet (admin only, premium all-around capabilities)", value="anthropic/claude-3.7-sonnet:beta"),
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
                if model.startswith("deepseek/"):
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
                
                if success:
                    # Create a description of all model strengths
                    model_descriptions = [
                        "**DeepSeek-R1**: Exceptional for roleplaying with vivid descriptions and strong character voice. Creates memorable responses with excellent metaphors. Best for immersion but may prioritize style over strict factual precision.",
                        "**Gemini 2.0 Flash**: Superior citation formatting and document analysis. Provides well-structured information, faster responses, and supports image analysis. Ideal for research but less immersive than other models.",
                        "**Nous: Hermes 405B Instruct**: Good balance between creativity and facts with strong reasoning. Handles complex topics with nuance while maintaining character. Perfect middle ground but not specialized in either direction.",
                        "**Claude 3.5 Haiku**: Fast, creative responses balancing efficiency and character. Supports image analysis with concise delivery. Good for quick interactions but less elaborate than larger models.",
                        "**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only)",
                        "**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only)"
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

        @self.tree.command(name="get_model", description="Show your currently selected AI model")
        async def get_model(interaction: discord.Interaction):
            await interaction.response.defer()
            try:
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )
                
                # Get friendly model name based on the model value
                model_name = "Unknown Model"
                if preferred_model.startswith("deepseek/"):
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
                
                # Create a description of all model strengths
                model_descriptions = [
                    "**DeepSeek-R1**: Better for roleplaying, more creative responses, and in-character immersion, but is slower to respond",
                    "**Gemini 2.0 Flash**: Better for accurate citations, factual responses, document analysis, image viewing capabilities, and has very fast response times",
                    "**Nous: Hermes 405B Instruct**: High reasoning capabilities, balanced between creativity and accuracy",
                    "**Claude 3.5 Haiku**: Excellent for comprehensive lore analysis and nuanced understanding, and has image viewing capabilities",
                    "**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only)",
                    "**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only)"
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
                
        @self.tree.command(name="query", description="Ask Publicia a question about Ledus Banum 77 and Imperial lore")
        @app_commands.describe(
            question="Your question about the lore",
            image_url="Optional URL to an image you want to analyze (must be a direct image URL ending with .jpg, .png, etc.)"
        )
        async def query_lore(interaction: discord.Interaction, question: str, image_url: str = None):
            await interaction.response.defer()
            try:
                if not question:
                    await interaction.followup.send("*neural error detected!* Please provide a question.")
                    return
                    
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
                                            await interaction.followup.send("*neural error detected!* The URL does not point to a valid image.", ephemeral=True)
                                            return
                                    else:
                                        await interaction.followup.send(f"*neural error detected!* Could not download image (status code: {resp.status}).", ephemeral=True)
                                        return
                        else:
                            await interaction.followup.send("*neural error detected!* The URL does not appear to be a direct image link. Please provide a URL ending with .jpg, .png, etc.", ephemeral=True)
                            return
                    except Exception as e:
                        logger.error(f"Error processing image URL: {e}")
                        await interaction.followup.send("*neural error detected!* Failed to process the image URL.", ephemeral=True)
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
                
                # Extract image IDs from search results
                image_ids = []
                for doc, chunk, score, image_id in search_results:
                    if image_id and image_id not in image_ids:
                        image_ids.append(image_id)
                        logger.info(f"Found relevant image: {image_id}")
                
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
                for doc, chunk, score, image_id in search_results:
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
                        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                        raw_doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
                    else:
                        raw_doc_contexts.append(f"From document '{doc}' (similarity: {score:.2f}):\n{chunk}")

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

                # Get user's preferred model
                preferred_model = self.user_preferences_manager.get_preferred_model(
                    str(interaction.user.id), 
                    default_model=self.config.LLM_MODEL
                )

                # Get friendly model name based on the model value
                model_name = "Unknown Model"
                if preferred_model.startswith("deepseek/"):
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

                if (image_attachments or image_ids) and preferred_model not in self.vision_capable_models:
                    await status_message.edit(content=f"*formulating one-off response with enhanced neural mechanisms using {model_name}...*\n*note: your preferred model ({model_name}) doesn't support image analysis. only the text content will be processed.*")
                    # No model switching - continues with user's preferred model
                else:
                    await status_message.edit(content=f"*formulating one-off response with enhanced neural mechanisms using {model_name}...*")

                # Step 5: Get AI response using user's preferred model
                

                completion, actual_model = await self._try_ai_completion(  # Now unpack the tuple of (completion, actual_model)
                    preferred_model,
                    messages,
                    image_ids=image_ids,
                    image_attachments=image_attachments,
                    temperature=0.1
                )

                if completion and completion.get('choices'):
                    response = completion['choices'][0]['message']['content']
                    
                    # No longer updating conversation history for query command
                    # This makes it a one-off interaction
                    
                    # Use the status message as the existing message for the first chunk
                    await self.send_split_message(
                        interaction.channel,
                        response,
                        model_used=actual_model,  # Pass the actual model used, not just the preferred model
                        user_id=str(interaction.user.id),
                        existing_message=None  # Don't use the ephemeral status message
                    )
                    
                    # Delete the status message since it's ephemeral and we've now sent the response
                    try:
                        await status_message.delete()
                    except:
                        pass
                else:
                    await interaction.followup.send("*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                await interaction.followup.send("*neural circuit overload!* My brain is struggling and an error has occurred.")

        @self.tree.command(name="search_docs", description="Search the document knowledge base")
        @app_commands.describe(query="What to search for")
        async def search_documents(interaction: discord.Interaction, query: str):
            await interaction.response.defer()
            try:
                if not query:
                    await interaction.followup.send("*neural error detected!* Please provide a search query.")
                    return
                    
                results = self.document_manager.search(query, top_k=10)
                if not results:
                    await interaction.followup.send("No relevant documents found.")
                    return
                response = "Search results:\n```"
                for doc_name, chunk, similarity, image_id in results:
                    if image_id:
                        # This is an image search result
                        image_name = self.image_manager.metadata[image_id]['name'] if image_id in self.image_manager.metadata else "Unknown Image"
                        response += f"\nIMAGE: {image_name} (ID: {image_id}, similarity: {similarity:.2f}):\n"
                        response += f"{chunk[:200]}...\n"
                    else:
                        response += f"\nFrom {doc_name} (similarity: {similarity:.2f}):\n"
                        response += f"{chunk[:200]}...\n"
                response += "```"
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
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
                
                # Knowledge Base
                response += "## **KNOWLEDGE BASE & LIMITATIONS**\n\n"
                response += "**📚 What I Know**\n"
                response += "• My knowledge is based on documents and images uploaded to my neural database\n"
                response += "• I specialize in Ledus Banum 77 (aka Tundra) lore and Imperial institutions\n"
                response += "• I can cite specific documents when providing information\n"
                response += "• I understand the Infinite Empire's structure, planes of existence, and Resonant Drilling\n\n"
                
                response += "**⚠️ What I Don't Know**\n"
                response += "• Information not contained in my document or image database\n"
                response += "• I cannot make up lore or information that isn't documented\n"
                response += "• I do not have knowledge about Earth or our real universe\n"
                response += "• I cannot access the internet or information outside my documents\n\n"
                
                # How I Work
                response += "## **HOW I WORK**\n\n"
                response += "**🧠 Neural Processing**\n"
                response += "• I use semantic search to find relevant information in my documents and images\n"
                response += "• I analyze your query to understand what you're looking for\n"
                response += "• I synthesize information from multiple documents when needed\n"
                response += "• I provide citations to document sources when possible\n"
                response += "• I use vector embeddings to match your questions with relevant content\n"
                response += "• I automatically extract content from Google Docs linked in your queries\n\n"
                
                # Image Analysis
                response += "**🖼️ Image Analysis**\n"
                response += "• I can analyze images in three ways:\n"
                response += "- Attach an image directly when mentioning me\n"
                response += "- Use `/query` with an image URL\n"
                response += "- I can search my image database for relevant visual information\n"
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
                response += "• `/view_image` - View an image from my knowledge base\n"
                response += "• `/remove_doc` - Remove a document from my knowledge base\n"
                response += "• `/remove_image` - Remove an image from my knowledge base\n"
                response += "• `/remove_googledoc` - Disconnect a Google Doc\n"
                response += "• `/rename_document` - Rename a document in my database\n"
                response += "• `/search_docs` - Search directly in my document knowledge base\n"
                response += "• `/update_image_description` - Update the description for an image\n\n"
                
                # Conversation Management
                response += "## **CONVERSATION MANAGEMENT**\n\n"
                response += "**💬 Conversation History**\n"
                response += "• `/history` - View your complete conversation history with me\n"
                response += "• `/manage_history` - View recent messages with numbered indices for deletion\n"
                response += "• `/delete_history_messages` - Remove specific messages by indices (use confirm:yes)\n"
                response += "• Type \"LOBOTOMISE\" in a message to Publicia, or use the /lobotomise command to wipe your entire conversation history\n"
                response += "• I remember our conversations to provide better context-aware responses\n\n"
                
                # Customization
                response += "## **CUSTOMIZATION**\n\n"
                response += "**⚙️ AI Model Selection**\n"
                response += "• `/set_model` - Choose your preferred AI model:\n"
                response += "- **DeepSeek-R1**: Best for immersive roleplaying and creative responses, but is slower to respond\n"
                response += "- **Gemini 2.0 Flash**: Best for accuracy, citations, and image analysis, and is very fast\n"
                response += "- **Nous: Hermes 405B**: Balanced between creativity and factual precision\n"
                response += "- **Claude 3.5 Haiku**: Fast responses with image capabilities\n"
                response += "- **Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only)\n"
                response += "- **Claude 3.7 Sonnet**: Admin only, premium all-around capabilities\n"
                response += "• `/get_model` - Check which model you're currently using\n"
                response += "• `/toggle_debug` - Show/hide which model generated each response\n\n"
                
                # Technical Information
                response += "## **TECHNICAL INFORMATION**\n\n"
                response += "**⚙️ Technical Details**\n"
                response += "• I'm powered by OpenRouter.ai with access to multiple LLM models\n"
                response += "• I have automatic fallback between models if one fails\n"
                response += "• I process documents and images using semantic search and vector embeddings\n"
                response += "• My database stores text chunks, image descriptions, and their embeddings\n"
                response += "• Google Doc integration uses public access to fetch document content\n"
                response += "• Image analysis requires vision-capable models (Gemini, Claude)\n\n"
                
                # Tips
                response += "## **TIPS FOR BEST RESULTS**\n\n"
                response += "• Ask specific questions for more accurate answers\n"
                response += "• If I don't know something, add relevant documents or images to my database\n"
                response += "• Use Google Docs integration for large, regularly updated documents\n"
                response += "• Include links to Google Docs in your queries for on-the-fly context\n"
                response += "• For adding images, I recommend labelling things within the image to help me have a better idea of how it relates to the lore\n"
                response += "• If you're unsure about the model to use, use the default Gemini 2.0 Flash for general queries\n"
                response += "*my genetically enhanced brain is always ready to help... just ask!*"
                
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
    
    async def _try_ai_completion(self, model: str, messages: List[Dict], image_ids=None, image_attachments=None, **kwargs) -> Tuple[Optional[Any], Optional[str]]:
        """Get AI completion with dynamic fallback options based on the requested model."""
        
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
                "deepseek/deepseek-r1:floor",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-r1:nitro",
                "deepseek/deepseek-r1",
                "deepseek/deepseek-chat",
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-distill-qwen-32b"
            ]
            models.extend([fb for fb in fallbacks if fb not in models])
        elif model_family == "google":
            fallbacks = [
                "google/gemini-2.0-flash-thinking-exp:free",
                "google/gemini-2.0-pro-exp-02-05:free",
                "google/gemini-2.0-flash-001"
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
        
        # Add general fallbacks
        general_fallbacks = [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-chat",
            "google/gemini-2.0-pro-exp-02-05:free",
            "nousresearch/hermes-3-llama-3.1-405b",
            "anthropic/claude-3.5-haiku:beta",
            "anthropic/claude-3.5-haiku"
        ]
        
        # Add general fallbacks that aren't already in the list
        for fb in general_fallbacks:
            if fb not in models:
                models.append(fb)
        
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
                                logger.error(f"API error (Status {response.status}): {error_text}")
                                # Log additional context like headers to help diagnose issues
                                logger.error(f"Request context: URL={response.url}, Headers={response.headers}")
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
                        
                    return completion, current_model  # Return both the completion and the model used
                    
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
            
            # Check if the stripped question is empty
            if not question:
                question = "Hello"
                logger.info("Received empty message after stripping mentions, defaulting to 'Hello'")
            
            # Check for Google Doc links in the message
            google_doc_ids = await self._extract_google_doc_ids(question)
            google_doc_contents = []
            
            if google_doc_ids:
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

            # Extract image IDs from search results
            image_ids = []
            for doc, chunk, score, image_id in search_results:
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
                    logger.info(f"Found relevant image: {image_id}")

            # Format raw results with citation info
            import urllib.parse
            raw_doc_contexts = []
            for doc, chunk, score, image_id in search_results:
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
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/edit?findtext={encoded_search}"
                    raw_doc_contexts.append(f"From document '{doc}' [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
                else:
                    raw_doc_contexts.append(f"From document '{doc}' (similarity: {score:.2f}):\n{chunk}")

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

            # Get user's preferred model
            preferred_model = self.user_preferences_manager.get_preferred_model(
                str(message.author.id), 
                default_model=self.config.LLM_MODEL
            )

            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            if preferred_model.startswith("deepseek/"):
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

            # Add a note about vision capabilities if relevant
            if (image_attachments or image_ids) and preferred_model not in self.vision_capable_models:
                await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(note: your preferred model ({model_name}) doesn't support image analysis. only the text content will be processed)*")
            else:
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(fetched content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})*")
                else:
                    await thinking_msg.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")
            
            # Step 5: Get AI response using user's preferred model
            completion, actual_model = await self._try_ai_completion(
                preferred_model,
                messages,
                image_ids=image_ids,
                image_attachments=image_attachments,
                temperature=0.1
            )

            if completion and completion.get('choices'):
                response = completion['choices'][0]['message']['content']
                
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    doc_note = f"\n\n*Note: I've included content from {len(google_doc_contents)} Google Doc{'s' if len(google_doc_contents) > 1 else ''} linked in your message.*"
                    response += doc_note
                
                # Add a note about found images if any were found in search results
                if image_ids:
                    img_note = f"\n\n*Note: I found {len(image_ids)} relevant image{'s' if len(image_ids) > 1 else ''} in my knowledge base related to your query.*"
                    response += img_note
                
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
"""
Document management, embeddings, and search functionality for Publicia
"""
import os
import re
import json
import pickle
import logging
import aiohttp
import asyncio
import urllib.parse
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from textwrap import shorten
from rank_bm25 import BM25Okapi
from typing import Optional # Added for type hinting

from utils.logging import sanitize_for_logging

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document storage, embeddings, and retrieval."""

    def _sanitize_name(self, name: str) -> str:
        """Sanitizes a string to be safe for use as a filename or dictionary key."""
        if not isinstance(name, str):
            name = str(name) # Ensure it's a string

        # Replace potentially problematic characters with underscores
        # Includes Windows reserved chars: <>:"/\|?*
        # Also includes control characters (0-31)
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)

        # Replace leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Collapse multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Ensure the name is not empty after sanitization
        if not sanitized:
            sanitized = "untitled"
        # Optional: Limit length?
        # max_len = 200
        # sanitized = sanitized[:max_len]
        return sanitized

    def _get_original_name(self, sanitized_name: str) -> str:
        """Retrieve the original document name from metadata given the sanitized name."""
        if sanitized_name in self.metadata:
            return self.metadata[sanitized_name].get('original_name', sanitized_name)
        return sanitized_name # Fallback if not found (shouldn't happen often)

    def _get_sanitized_name_from_original(self, original_name: str) -> Optional[str]:
        """Find the sanitized name corresponding to an original name."""
        # First, try direct sanitization
        s_name_direct = self._sanitize_name(original_name)
        if s_name_direct in self.metadata:
             # Verify if the original name matches
             if self.metadata[s_name_direct].get('original_name') == original_name:
                 return s_name_direct

        # If direct match fails, iterate through metadata (slower fallback)
        for s_name, meta in self.metadata.items():
            if meta.get('original_name') == original_name:
                return s_name
        logger.warning(f"Could not find sanitized name for original name: {original_name}")
        return None # Not found
    
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
        self.contextualized_chunks: Dict[str, List[str]] = {}
        
        # BM25 indexes for text search
        self.bm25_indexes = {}
        
        # Load existing documents
        logger.info("Starting document loading")
        # Actual loading is done asynchronously in _load_documents()
        # For now we just initialize with empty data
        logger.info("Document loading will be done asynchronously")

        # Define the name for the internal document list
        self._internal_list_doc_name = "_internal_document_list.txt"


    async def _update_document_list_file(self):
        """Creates or updates a special document listing all other documents."""
        logger.info("Updating internal document list file...")
        try:
            # Get original document names from metadata, excluding the list file itself
            s_internal_list_name = self._sanitize_name(self._internal_list_doc_name)
            original_doc_names = sorted([
                self._get_original_name(s_name) for s_name in self.metadata.keys()
                if s_name != s_internal_list_name
            ])

            # Format the content using original names
            if not original_doc_names:
                content = "No documents are currently managed."
            else:
                content = "Managed Documents:\n\n" + "\n".join(f"- {orig_name}" for orig_name in original_doc_names)

            # Check if the document exists and if content needs updating
            # Use sanitized name for file path
            needs_update = True
            list_doc_path = self.base_dir / s_internal_list_name
            if list_doc_path.exists():
                try:
                    with open(list_doc_path, 'r', encoding='utf-8-sig') as f:
                        existing_content = f.read()
                    if existing_content == content:
                        needs_update = False
                        logger.info("Internal document list content is already up-to-date.")
                except Exception as e:
                    logger.warning(f"Could not read existing internal document list file ({list_doc_path}): {e}. Will overwrite.")

            if needs_update:
                logger.info(f"Content for internal list doc ('{self._internal_list_doc_name}', sanitized: {s_internal_list_name}) requires update. Adding/updating document.")
                # Use add_document with the ORIGINAL name, it will handle sanitization internally
                # Pass _internal_call=True to prevent recursion
                success = await self.add_document(self._internal_list_doc_name, content, save_to_disk=True, _internal_call=True)
                if success:
                    logger.info(f"Successfully updated and saved internal list doc ('{self._internal_list_doc_name}', sanitized: {s_internal_list_name}).")
                else:
                    logger.error(f"Failed to add/update internal list doc ('{self._internal_list_doc_name}', sanitized: {s_internal_list_name}) due to errors during processing or saving.")
            else:
                 # Even if content didn't change, ensure metadata reflects check time
                 # Use sanitized name for lookup
                 if s_internal_list_name in self.metadata:
                     self.metadata[s_internal_list_name]['checked'] = datetime.now().isoformat()
                     # No need to save just for a timestamp check unless other changes happened

        except Exception as e:
            logger.error(f"Failed to update internal document list file: {e}")
            import traceback
            logger.error(traceback.format_exc())


    async def _get_google_embedding_async(self, text: str, task_type: str, title: Optional[str] = None) -> Optional[np.ndarray]:
        """Helper function to asynchronously get a single embedding from Google API."""
        if not text or not text.strip():
            logger.warning("Skipping empty text for embedding")
            return None

        api_key = self.config.GOOGLE_API_KEY
        if not api_key:
            logger.error("Google API Key not configured for async embedding.")
            return None

        # Construct the API URL
        # The self.embedding_model variable already contains the "models/" prefix.
        api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.embedding_model}:embedContent?key={api_key}"

        payload = {
            "content": {"parts": [{"text": text}]},
            "task_type": task_type,
        }
        if title and task_type == "retrieval_document":
            payload["title"] = title

        headers = {"Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "embedding" in result and "values" in result["embedding"]:
                            embedding_vector = np.array(result["embedding"]["values"])
                            # Truncate if dimensions is specified
                            if self.embedding_dimensions and self.embedding_dimensions < len(embedding_vector):
                                embedding_vector = embedding_vector[:self.embedding_dimensions]
                            return embedding_vector
                        else:
                            logger.error(f"Unexpected embedding response structure: {result}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Google Embedding API error (Status {response.status}): {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.error("Timeout calling Google Embedding API.")
            return None
        except Exception as e:
            logger.error(f"Error calling Google Embedding API: {e}")
            return None

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
        # Iterate using sanitized names (s_name)
        for s_name, chunks in self.chunks.items():
            # Make sure we have a BM25 index for this document using sanitized name
            if s_name not in self.bm25_indexes:
                self.bm25_indexes[s_name] = self._create_bm25_index(
                    self.contextualized_chunks.get(s_name, chunks) # Use s_name for lookup
                )
                
            # Get BM25 scores using sanitized name
            bm25_scores = self.bm25_indexes[s_name].get_scores(query_tokens)
            
            # Add top results
            for idx, score in enumerate(bm25_scores):
                if idx < len(chunks):  # Safety check
                    image_id = None
                    # Check image prefix using sanitized name
                    if s_name.startswith("image_") and s_name.endswith(".txt"):
                        image_id = s_name[6:-4]
                    # Check metadata using sanitized name
                    elif s_name in self.metadata and 'image_id' in self.metadata[s_name]:
                        image_id = self.metadata[s_name]['image_id']
                    
                    # Use contextualized chunk instead of original, using sanitized name
                    chunk = self.get_contextualized_chunk(s_name, idx)
                    
                    # Get the original name for the result tuple
                    original_name = self._get_original_name(s_name)
                    
                    results.append((
                        original_name, # Return original name
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
        """Get the contextualized chunk if available, otherwise fall back to the original chunk."""
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
        # Assume embedding_results contains ORIGINAL names, need to map to sanitized for key
        for original_name, chunk, score, image_id, chunk_idx, total_chunks in embedding_results:
            s_name = self._get_sanitized_name_from_original(original_name)
            if not s_name:
                logger.warning(f"Could not find sanitized name for '{original_name}' in embedding results, skipping.")
                continue
            key = (s_name, chunk_idx) # Use sanitized name in key
            # Ensure score is between 0 and 1
            norm_score = max(0, min(1, score))
            combined_scores[key] = combined_scores.get(key, 0) + (norm_score * embedding_weight)

            # Log some of the top embedding scores (using original name for logging)
            if len(combined_scores) <= 3:
                logger.info(f"Top embedding result: {original_name}, score: {score:.4f}, normalized: {norm_score:.4f}")
        
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
        # Assume bm25_results contains ORIGINAL names, need to map to sanitized for key
        for original_name, chunk, score, image_id, chunk_idx, total_chunks in bm25_results:
            s_name = self._get_sanitized_name_from_original(original_name)
            if not s_name:
                logger.warning(f"Could not find sanitized name for '{original_name}' in BM25 results, skipping.")
                continue
            key = (s_name, chunk_idx) # Use sanitized name in key
            # Normalize BM25 score to [0, 1] range
            if range_bm25 > 0:
                norm_score = (score - min_bm25) / range_bm25
            else:
                norm_score = 0.5  # Default if all scores are the same

            combined_scores[key] = combined_scores.get(key, 0) + (norm_score * bm25_weight)

            # Log some of the top BM25 scores (using original name for logging)
            #if len([k for k in combined_scores.keys() if k == key]) <= 3:
            #    logger.info(f"Top BM25 result: {original_name}, score: {score:.4f}, normalized: {norm_score:.4f}")
        
        # Safety check - ensure we have some scores
        if not combined_scores:
            logger.warning("No combined scores found. Returning empty results.")
            return []
        
        # Create combined results
        combined_results = []
        # Iterate using sanitized names (s_name) from the combined_scores keys
        for (s_name, chunk_idx), score in combined_scores.items():
            # Use contextualized chunk instead of original, using sanitized name
            chunk_index = chunk_idx - 1  # Convert from 1-based to 0-based indexing

            # Safety check for valid index using sanitized name
            if chunk_index < 0 or chunk_index >= len(self.chunks.get(s_name, [])):
                logger.warning(f"Invalid chunk index {chunk_index} for sanitized doc '{s_name}' during combine.")
                continue

            # Get the contextualized chunk using sanitized name
            chunk = self.get_contextualized_chunk(s_name, chunk_index)

            image_id = None
            # Check image prefix using sanitized name
            if s_name.startswith("image_") and s_name.endswith(".txt"):
                image_id = s_name[6:-4]
            # Check metadata using sanitized name
            elif s_name in self.metadata and 'image_id' in self.metadata[s_name]:
                image_id = self.metadata[s_name]['image_id']

            # Get the original name for the final result tuple
            original_name = self._get_original_name(s_name)

            combined_results.append((
                original_name, # Return original name
                chunk,
                score,  # This is the combined score
                image_id,
                chunk_idx,
                len(self.chunks[s_name]) # Use sanitized name for chunk count lookup
            ))
        
        # Sort by score and return top_k
        combined_results.sort(key=lambda x: x[2], reverse=True)

        # Log top combined results (using original name)
        for i, (original_name, _, score, _, _, _) in enumerate(combined_results[:3]):
            logger.info(f"Top {i+1} combined result: {original_name}, score: {score:.4f}")
        
        return combined_results[:top_k]

    async def generate_chunk_context(self, document_content: str, chunk_content: str) -> str:
        """Generate context for a chunk using a list of fallback models via OpenRouter."""
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
            
            # List of models to try in order
            fallback_models = [
                #"google/gemini-2.5-pro-exp-03-25:free", # Added new model
                #"google/gemini-2.0-flash-thinking-exp:free",
                #"google/gemini-2.0-flash-exp:free",
                #"google/gemma-3-27b-it:free",
                "cohere/command-r7b-12-2024",
                "microsoft/phi-4-multimodal-instruct",
                "amazon/nova-micro-v1",
                "qwen/qwen-turbo",
                "google/gemma-3-12b-it",
                "google/gemini-2.0-flash-lite-001",
                "google/gemini-2.0-flash-001", # Fixed formatting
                "google/gemini-flash-1.5-8b",
            ]
            
            # Try each model in sequence until one works
            for model in fallback_models:
                logger.info(f"Attempting to generate chunk context with model: {model}")
                
                # Prepare payload with current model
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,  # Low temperature for deterministic outputs
                    "max_tokens": 150    # Limit token length
                }
                
                # Make API call with current model
                max_retries = 2  # Fewer retries per model since we have multiple models
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
                                    logger.error(f"API error with model {model} (Status {response.status}): {error_text}")
                                    # If we had a non-200 status, try again or continue to next model
                                    if attempt == max_retries - 1:
                                        # Try next model
                                        break
                                    continue
                                    
                                completion = await response.json()
                        
                        # Extract context from the response
                        if completion and completion.get('choices') and len(completion['choices']) > 0:
                            if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                                context = completion['choices'][0]['message']['content'].strip()
                                
                                logger.info(f"Successfully generated context with model {model} ({len(context.split())} words): {context[:50]}...")
                                return context
                        
                        # If we couldn't extract a valid context, try next model
                        logger.warning(f"Failed to extract valid context from model {model}")
                        break
                        
                    except Exception as e:
                        logger.error(f"Error generating chunk context with model {model} (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            # If we've used all retries for this model, try next model
                            break
                        # Otherwise, continue to next retry with same model
                        await asyncio.sleep(1)  # Wait before retrying
            
            # If all models failed, return default context
            logger.error("All models failed to generate context")
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

    async def generate_embeddings(self, texts: List[str], is_query: bool = False, titles: List[str] = None) -> np.ndarray:
        """Asynchronously generate embeddings for a list of text chunks using Google's Generative AI via REST API."""
        embeddings_list = []
        task_type = "retrieval_query" if is_query else "retrieval_document"

        # Create tasks for each text embedding generation
        tasks = []
        valid_indices = [] # Keep track of indices for non-empty texts
        for i, text in enumerate(texts):
            if text and text.strip():
                title = titles[i] if titles and i < len(titles) and not is_query else None
                tasks.append(self._get_google_embedding_async(text, task_type, title))
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty text at index {i} for embedding")

        # Run tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process results and handle potential None values (errors or empty inputs)
        placeholder_embedding = None
        final_embeddings = [None] * len(texts) # Initialize with None for all original texts

        for i, result_embedding in enumerate(results):
            original_index = valid_indices[i]
            if result_embedding is not None:
                final_embeddings[original_index] = result_embedding
                if placeholder_embedding is None:
                    placeholder_embedding = np.zeros_like(result_embedding) # Create placeholder based on first success
            else:
                # Error occurred for this text, will use placeholder later
                logger.warning(f"Failed to get embedding for text at index {original_index}, will use placeholder.")

        # Fill in placeholders for failed or empty texts
        if placeholder_embedding is None and any(e is None for e in final_embeddings):
             # If all failed and we couldn't create a placeholder, raise error or return empty
             logger.error("Failed to generate any embeddings and couldn't create placeholder.")
             # Depending on desired behavior, either raise or return empty array
             # raise ValueError("Failed to generate any embeddings.")
             return np.array([]) # Return empty array if all fail

        for i in range(len(texts)):
            if final_embeddings[i] is None:
                if placeholder_embedding is not None:
                    final_embeddings[i] = placeholder_embedding
                else:
                    # This case should ideally not happen if placeholder_embedding logic is correct
                    # If it does, it means even the placeholder failed. Maybe return empty or raise.
                    logger.error(f"Could not provide a placeholder for failed/empty embedding at index {i}")
                    # Decide handling: return empty array, raise error, etc.
                    return np.array([])


        return np.array(final_embeddings)


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

    async def add_document(self, name: str, content: str, save_to_disk: bool = True, _internal_call: bool = False) -> bool:
        """
        Add a new document to the system with contextual retrieval.

        Args:
            name (str): The name of the document.
            content (str): The content of the document.
            save_to_disk (bool): Whether to save changes immediately. Defaults to True.
            _internal_call (bool): Flag to prevent recursive calls when updating the internal list doc. Defaults to False.

        Returns:
            bool: True if the document was added/updated successfully (including saving if requested), False otherwise.
        """
        try:
            original_name = name # Keep original name for reference and metadata
            s_name = self._sanitize_name(name)

            if s_name != original_name:
                logger.warning(f"Document name '{original_name}' sanitized to '{s_name}' for internal storage and filename.")

            # Prevent direct modification of the internal list doc via this method if not internal call
            # Use sanitized name for check
            if s_name == self._sanitize_name(self._internal_list_doc_name) and not _internal_call:
                 logger.warning(f"Attempted to directly modify internal document '{original_name}' (sanitized: {s_name}). Use specific commands or let the system manage it.")
                 # Optionally, raise an error or return False
                 # raise ValueError(f"Cannot directly modify internal document '{name}'")
                 return False # Indicate failure

            if not content or not content.strip():
                logger.warning(f"Document {name} has no content. Skipping.")
                return True # Technically not a failure, just skipped

            # Calculate word count
            word_count = len(content.split())
            max_words_for_context = 20000 # Define the limit
            logger.info(f"Document '{original_name}' (sanitized: {s_name}) word count: {word_count}")

            # Create original chunks (always needed)
            chunks = self._chunk_text(content)
            if not chunks:
                logger.warning(f"Document '{original_name}' (sanitized: {s_name}) has no content to chunk. Skipping.")
                return True # Technically not a failure, just skipped

            # Check if document already exists and if content has changed
            content_changed = True
            current_hash = None # Initialize hash variable
            # Use sanitized name for lookup
            if s_name in self.metadata and 'content_hash' in self.metadata[s_name]:
                import hashlib
                current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                previous_hash = self.metadata[s_name]['content_hash']
                if previous_hash and previous_hash == current_hash:
                    content_changed = False
                    logger.info(f"Document '{original_name}' (sanitized: {s_name}) content has not changed based on hash.")

            # Determine if processing (embedding, indexing) is needed
            # Needs processing if content changed OR if embeddings are missing for this doc
            needs_processing = content_changed or s_name not in self.embeddings

            if needs_processing:
                logger.info(f"Processing required for document '{original_name}' (sanitized: {s_name}) (Content changed: {content_changed}, Embeddings missing: {s_name not in self.embeddings})")

                # Decide whether to contextualize based on word count
                should_contextualize = word_count <= max_words_for_context
                # logger.debug(f"Contextualization check for '{name}': word_count={word_count}, max_words={max_words_for_context}, should_contextualize={should_contextualize}") # DEBUG LOG REMOVED

                if should_contextualize:
                    logger.info(f"Document '{original_name}' ({word_count} words) is within limit ({max_words_for_context}). Generating contextualized chunks.")
                    contextualized_chunks = await self.contextualize_chunks(original_name, content, chunks) # Pass original_name for context generation title?
                    # Use contextualized chunks for embeddings and BM25
                    chunks_for_embedding = contextualized_chunks
                    chunks_for_bm25 = contextualized_chunks
                    # Store the contextualized chunks using sanitized name
                    self.contextualized_chunks[s_name] = contextualized_chunks
                else:
                    logger.warning(f"Document '{original_name}' ({word_count} words) exceeds limit ({max_words_for_context}). Skipping contextualization, using original chunks.")
                    # Use original chunks for embeddings and BM25
                    chunks_for_embedding = chunks
                    chunks_for_bm25 = chunks
                    # Ensure no stale contextualized chunks exist if the doc previously fit
                    if s_name in self.contextualized_chunks:
                        del self.contextualized_chunks[s_name]
                        logger.info(f"Removed previous contextualized chunks for oversized document '{original_name}' (sanitized: {s_name}).")

                # Generate embeddings using the selected chunks (either original or contextualized)
                # Ensure chunks_for_embedding is not empty before proceeding
                if not chunks_for_embedding:
                    logger.error(f"No chunks available for embedding document '{original_name}' (sanitized: {s_name}). Skipping embedding generation.")
                    embeddings = np.array([]) # Assign empty array if no chunks
                else:
                    # Use original_name for titles passed to embedding model
                    titles = [original_name] * len(chunks_for_embedding) # Title count should match chunk count
                    embeddings = await self.generate_embeddings(chunks_for_embedding, is_query=False, titles=titles)

                # Store document data using sanitized name as key
                self.chunks[s_name] = chunks  # Store original chunks
                self.embeddings[s_name] = embeddings # Store the generated embeddings

                # Calculate hash if not already done (e.g., if content changed but wasn't in metadata before)
                if current_hash is None:
                     import hashlib
                     current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                # Update metadata using sanitized name as key
                self.metadata[s_name] = {
                    'original_name': original_name, # Store original name
                    'added': self.metadata.get(s_name, {}).get('added', datetime.now().isoformat()), # Preserve original add time using sanitized key lookup
                    'updated': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'content_hash': current_hash,
                    'contextualized': should_contextualize # Add flag indicating if contextualized
                }

                # Create BM25 index using the selected chunks and sanitized name
                # Ensure chunks_for_bm25 is not empty
                if chunks_for_bm25:
                    self.bm25_indexes[s_name] = self._create_bm25_index(chunks_for_bm25) # Use sanitized name
                else:
                    logger.warning(f"No chunks available for BM25 indexing document '{original_name}' (sanitized: {s_name}). Skipping BM25 index creation.")
                    if s_name in self.bm25_indexes: # Remove stale index if it exists
                        del self.bm25_indexes[s_name]


                logger.info(f"Finished processing for document '{original_name}' (sanitized: {s_name}). Contextualized: {should_contextualize}")

            else: # Content hasn't changed, just update timestamp
                logger.info(f"Document '{original_name}' (sanitized: {s_name}) content unchanged. Updating 'checked' timestamp.")
                if s_name in self.metadata:
                    self.metadata[s_name]['checked'] = datetime.now().isoformat()
                else: # Should not happen if content_changed is False, but safety check
                     # If metadata was missing, create it with original name
                     self.metadata[s_name] = {
                        'original_name': original_name,
                        'checked': datetime.now().isoformat(),
                        'chunk_count': len(chunks) # Best guess
                     }

            # Save to disk if requested
            if save_to_disk:
                self._save_to_disk() # This will raise exceptions if it fails

            # Log completion message
            if needs_processing:
                 log_context_status = self.metadata.get(s_name, {}).get('contextualized', 'N/A')
                 logger.info(f"{'Internally added/updated' if _internal_call else 'Added/updated'} document: '{original_name}' (sanitized: {s_name}) with {len(chunks)} chunks. Contextualized: {log_context_status}")
            else:
                 logger.info(f"Document '{original_name}' (sanitized: {s_name}) verified unchanged")

            # Update the document list file, unless this was an internal call
            if not _internal_call:
                await self._update_document_list_file() # This needs updating too

            return True # Indicate success

        except Exception as e:
            logger.error(f"Error adding document {name}: {e}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback
            # Avoid raising if it's an internal call failing, maybe log differently?
            # if not _internal_call: # Keep the original logic comment but don't raise here
            #     raise
            return False # Indicate failure


    def get_googledoc_id_mapping(self):
        """Get mapping from document names to google doc IDs."""
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

    def get_all_document_contents(self) -> Dict[str, str]:
        """
        Retrieves the full content of all managed text documents.

        Returns:
            Dict[str, str]: A dictionary where keys are *original* document names
                            and values are the full document content.
                            Excludes image description files and internal lists.
        """
        all_contents = {}
        logger.info("Retrieving content for all managed text documents...")
        # Iterate using sanitized names (s_name)
        s_internal_list_name = self._sanitize_name(self._internal_list_doc_name)
        for s_name in self.metadata.keys():
            # Skip internal list file and image description files using sanitized name
            if s_name == s_internal_list_name or \
               (s_name.startswith("image_") and s_name.endswith(".txt")):
                logger.debug(f"Skipping non-content document (sanitized): {s_name}")
                continue

            original_name = self._get_original_name(s_name) # Get original name for the key
            file_path = self.base_dir / s_name # Use sanitized name for file path
            content = None
            if file_path.exists() and file_path.is_file():
                try:
                    # Use utf-8-sig to handle potential BOM
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Error reading content for document '{original_name}' (sanitized: {s_name}): {e}")
            else:
                logger.warning(f"Document file not found for '{original_name}' (sanitized: {s_name}), attempting to reconstruct from chunks.")
                # Fallback: Reconstruct from original chunks if file missing, using sanitized name
                if s_name in self.chunks:
                    content = " ".join(self.chunks[s_name])
                else:
                     logger.error(f"Could not retrieve content for '{original_name}' (sanitized: {s_name}) from file or chunks.")

            if content is not None:
                 all_contents[original_name] = content # Use original name as key

        logger.info(f"Retrieved content for {len(all_contents)} documents.")
        return all_contents

    async def search(self, query: str, top_k: int = None, apply_reranking: bool = None) -> List[Tuple[str, str, float, Optional[str], int, int]]: # Make async
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

            # Generate query embedding asynchronously
            query_embedding_result = await self.generate_embeddings([query], is_query=True) # Await async call
            if query_embedding_result.size == 0:
                 logger.error("Failed to generate query embedding.")
                 return []
            query_embedding = query_embedding_result[0]


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
                return await self.rerank_results(query, combined_results, top_k=top_k) # Await async call
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

        # Iterate using sanitized names (s_name)
        for s_name, doc_embeddings in self.embeddings.items():
            if len(doc_embeddings) == 0:
                logger.warning(f"Skipping document '{self._get_original_name(s_name)}' (sanitized: {s_name}) with empty embeddings")
                continue

            # Calculate similarities (dot product since embeddings are normalized)
            similarities = np.dot(doc_embeddings, query_embedding)

            if len(similarities) > 0:
                top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):]

                for idx in top_indices:
                    image_id = None
                    # Check image prefix using sanitized name
                    if s_name.startswith("image_") and s_name.endswith(".txt"):
                        image_id = s_name[6:-4]
                    # Check metadata using sanitized name
                    elif s_name in self.metadata and 'image_id' in self.metadata[s_name]:
                        image_id = self.metadata[s_name]['image_id']

                    # Check chunk index validity using sanitized name
                    if idx < len(self.chunks[s_name]):
                        # Use contextualized chunk instead of original, using sanitized name
                        chunk = self.get_contextualized_chunk(s_name, idx)

                        # Get the original name for the result tuple
                        original_name = self._get_original_name(s_name)

                        results.append((
                            original_name, # Return original name
                            chunk,
                            float(similarities[idx]),
                            image_id,
                            idx + 1,  # 1-based indexing for display
                            len(self.chunks[s_name]) # Use sanitized name for chunk count
                        ))
        
        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)

        # Log search results (using original name)
        for original_name, chunk, similarity, image_id, chunk_index, total_chunks in results[:top_k]:
            logger.info(f"Found relevant chunk in {original_name} (similarity: {similarity:.2f}, chunk: {chunk_index}/{total_chunks})")
            if image_id:
                logger.info(f"This is an image description for image ID: {image_id}")

            # Log whether we're using a contextualized chunk (needs sanitized name for lookup)
            s_name = self._get_sanitized_name_from_original(original_name)
            if s_name:
                is_contextualized = (s_name in self.contextualized_chunks and
                                    chunk_index - 1 < len(self.contextualized_chunks[s_name]) and
                                    chunk == self.contextualized_chunks[s_name][chunk_index - 1])
                #logger.info(f"Using {'contextualized' if is_contextualized else 'original'} chunk for {original_name}")
            else:
                #logger.info(f"Could not verify contextualization status for {original_name}")
                pass # Avoid logging if lookup failed

            #logger.info(f"Chunk content: {shorten(sanitize_for_logging(chunk), width=300, placeholder='...')}")

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
            #with open(self.base_dir / 'embeddings_provider.txt', 'w') as f:
            #    f.write("gemini")
                
        except PermissionError as pe:
            logger.error(f"Permission denied error saving data to {self.base_dir}. Check write permissions for the directory and its contents (e.g., embeddings.pkl). Error: {pe}")
            raise # Re-raise after specific logging
        except Exception as e:
            logger.error(f"Error saving document data to disk: {e}")
            raise # Re-raise other exceptions

    async def regenerate_all_embeddings(self): # Make async
        """Regenerate all embeddings using the current embedding model."""
        try:
            logger.info("Starting regeneration of all embeddings")

            # Iterate through all documents using sanitized names (s_name)
            for s_name, chunks in self.chunks.items():
                original_name = self._get_original_name(s_name)
                logger.info(f"Regenerating embeddings for document: '{original_name}' (sanitized: {s_name})")

                # Get contextualized chunks (regenerate if needed, although ideally they exist)
                # Use sanitized name for lookups
                if s_name not in self.contextualized_chunks:
                     logger.warning(f"Contextualized chunks missing for '{original_name}' (sanitized: {s_name}), regenerating...")
                     doc_path = self.base_dir / s_name # Use sanitized name for path
                     if doc_path.exists():
                         with open(doc_path, 'r', encoding='utf-8-sig') as f:
                             doc_content = f.read()
                         # Use original name for context generation title?
                         self.contextualized_chunks[s_name] = await self.contextualize_chunks(original_name, doc_content, chunks)
                     else:
                         logger.error(f"Cannot find original file '{original_name}' (sanitized: {s_name}) to regenerate contextualized chunks.")
                         continue # Skip this document if original file is missing

                # Use sanitized name for lookup, fallback to original chunks
                contextualized_chunks_for_doc = self.contextualized_chunks.get(s_name, chunks)

                # Generate new embeddings asynchronously
                # Pass ORIGINAL name as title
                titles = [original_name] * len(contextualized_chunks_for_doc)
                new_embeddings = await self.generate_embeddings(contextualized_chunks_for_doc, is_query=False, titles=titles) # Await async call

                # Update stored embeddings using sanitized name
                self.embeddings[s_name] = new_embeddings

            # Save to disk
            self._save_to_disk()

            logger.info("Completed regeneration of all embeddings")
            return True
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}")
            return False

    async def _load_documents(self, force_reload: bool = False):
        """Load document data from disk and add any new .txt files."""
        # Wrap the entire loading process in a try/except
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

                    # Regenerate embeddings for all documents using sanitized names (s_name)
                    for s_name, chunks in self.chunks.items():
                        original_name = self._get_original_name(s_name) # Get original name early
                        logger.info(f"Regenerating embeddings for document: '{original_name}' (sanitized: {s_name}) due to provider change")

                        # Get original document content if available using sanitized name for path
                        doc_path = self.base_dir / s_name
                        if doc_path.exists():
                            with open(doc_path, 'r', encoding='utf-8-sig') as f:
                                doc_content = f.read()
                        else:
                            # If original not available, concatenate chunks
                            logger.warning(f"Original file not found for '{original_name}' (sanitized: {s_name}), using concatenated chunks.")
                            doc_content = " ".join(chunks)

                        # Generate contextualized chunks using original name for title?
                        contextualized_chunks = await self.contextualize_chunks(original_name, doc_content, chunks)
                        self.contextualized_chunks[s_name] = contextualized_chunks # Store with sanitized key

                        # Generate embeddings asynchronously using ORIGINAL name as title
                        titles = [original_name] * len(contextualized_chunks)
                        self.embeddings[s_name] = await self.generate_embeddings(contextualized_chunks, is_query=False, titles=titles) # Store with sanitized key

                        # Create BM25 index using sanitized key
                        self.bm25_indexes[s_name] = self._create_bm25_index(contextualized_chunks)

                        # Update metadata using sanitized key, store original name
                        self.metadata[s_name] = {
                            'original_name': original_name,
                            'added': datetime.now().isoformat(), # Reset added time? Or try to preserve?
                            'updated': datetime.now().isoformat(),
                            'chunk_count': len(chunks),
                            'content_hash': self.metadata.get(s_name, {}).get('content_hash'), # Preserve hash if possible
                            'contextualized': True # Assume contextualized after regeneration
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
                            logger.warning(f"Contextualized chunks missing for existing document '{doc_name}' during BM25 index rebuild. Checking word count before regenerating.")
                            # Generate contextualized chunks only if within limit
                            doc_path = self.base_dir / doc_name
                            doc_content = ""
                            if doc_path.exists():
                                try:
                                    with open(doc_path, 'r', encoding='utf-8-sig') as f:
                                        doc_content = f.read()
                                except Exception as e:
                                     logger.error(f"Error reading document file {doc_name} for contextualization check: {e}")
                                     # Fallback to original chunks if file read fails
                                     self.bm25_indexes[doc_name] = self._create_bm25_index(chunks)
                                     continue # Skip to next document

                            else:
                                # If original file not available, concatenate chunks for word count check
                                logger.warning(f"Original file {doc_name} not found. Using concatenated chunks for word count check.")
                                doc_content = " ".join(chunks)

                            # Perform the word count check here
                            word_count = len(doc_content.split())
                            max_words_for_context = 20000 # Use the same limit
                            should_contextualize_rebuild = word_count <= max_words_for_context
                            logger.info(f"BM25 rebuild check for '{doc_name}': word_count={word_count}, max_words={max_words_for_context}, should_contextualize={should_contextualize_rebuild}")

                            if should_contextualize_rebuild:
                                logger.info(f"Document '{doc_name}' is within limit. Regenerating contextualized chunks for BM25 index.")
                                contextualized_chunks = await self.contextualize_chunks(doc_name, doc_content, chunks)
                                self.contextualized_chunks[doc_name] = contextualized_chunks
                                self.bm25_indexes[doc_name] = self._create_bm25_index(contextualized_chunks)
                                # Update metadata flag if possible
                                if doc_name in self.metadata:
                                    self.metadata[doc_name]['contextualized'] = True
                            else:
                                logger.warning(f"Document '{doc_name}' exceeds limit ({word_count} words). Using original chunks for BM25 index rebuild.")
                                self.bm25_indexes[doc_name] = self._create_bm25_index(chunks)
                                # Ensure no stale contextualized chunks exist
                                if doc_name in self.contextualized_chunks:
                                    del self.contextualized_chunks[doc_name]
                                # Update metadata flag if possible
                                if doc_name in self.metadata:
                                    self.metadata[doc_name]['contextualized'] = False
                    
                    logger.info(f"Loaded {len(self.chunks)} documents from processed data")
            else:
                # Start fresh if force_reload is True or no processed data exists
                self.chunks = {}
                self.contextualized_chunks = {}
                self.embeddings = {}
                self.metadata = {}
            self.bm25_indexes = {}
            logger.info("Starting fresh or force reloading documents")

            # --- Migration Step for Existing Data ---
            migrated_count = 0
            keys_to_migrate = list(self.metadata.keys()) # Iterate over a copy of keys
            needs_save_after_migration = False

            logger.info("Checking existing document metadata for necessary migrations...")
            for current_key in keys_to_migrate:
                # Add a try-except block for individual entry migration
                try:
                    meta = self.metadata.get(current_key)
                    if not meta:
                        continue # Skip if metadata somehow missing for this key

                    original_name_in_meta = meta.get('original_name')
                    # Calculate what the key *should* be if the current_key was the original name
                    expected_sanitized_key_if_current_is_orig = self._sanitize_name(current_key)

                    # Scenario 1: Key is unsanitized (doesn't match its sanitized version)
                    # Scenario 2: Key is sanitized, but 'original_name' field is missing (old format)
                    needs_migration = False
                    original_name_to_set = None

                    if current_key != expected_sanitized_key_if_current_is_orig:
                        # Key itself is likely the old, unsanitized name
                        needs_migration = True
                        original_name_to_set = current_key # The key *is* the original name
                        logger.warning(f"Found potentially unsanitized key '{current_key}'. Preparing migration.")
                    elif original_name_in_meta is None:
                        # Key is sanitized, but metadata is old format (missing original_name)
                        needs_migration = True
                        original_name_to_set = current_key # Assume the key was the intended original name
                        logger.warning(f"Found old metadata format for key '{current_key}' (missing 'original_name'). Preparing update.")

                    if needs_migration and original_name_to_set:
                        new_sanitized_key = self._sanitize_name(original_name_to_set)

                        # Check for conflict: If the new key exists and is not the current key
                        if new_sanitized_key in self.metadata and new_sanitized_key != current_key:
                            logger.error(f"Migration conflict! Cannot migrate '{current_key}' to '{new_sanitized_key}' because the target key already exists. Skipping migration for this entry.")
                            continue # Skip this problematic entry

                        logger.info(f"Migrating entry: '{current_key}' -> '{new_sanitized_key}' (Original: '{original_name_to_set}')")

                        # Move data to new key only if the key is actually changing
                        if new_sanitized_key != current_key:
                            if current_key in self.chunks: self.chunks[new_sanitized_key] = self.chunks.pop(current_key)
                            if current_key in self.contextualized_chunks: self.contextualized_chunks[new_sanitized_key] = self.contextualized_chunks.pop(current_key)
                            if current_key in self.embeddings: self.embeddings[new_sanitized_key] = self.embeddings.pop(current_key)
                            if current_key in self.bm25_indexes: self.bm25_indexes[new_sanitized_key] = self.bm25_indexes.pop(current_key)
                            # Move metadata last
                            meta_to_move = self.metadata.pop(current_key)
                            meta_to_move['original_name'] = original_name_to_set # Ensure original name is set
                            self.metadata[new_sanitized_key] = meta_to_move
                        else:
                            # Key isn't changing, just update metadata in place
                            self.metadata[current_key]['original_name'] = original_name_to_set

                        # Attempt to rename file on disk
                        old_file_path = self.base_dir / current_key # Path based on the old key
                        new_file_path = self.base_dir / new_sanitized_key # Path based on the new key
                        if old_file_path.exists() and old_file_path != new_file_path:
                            try:
                                old_file_path.rename(new_file_path)
                                logger.info(f"Successfully renamed file: {old_file_path} -> {new_file_path}")
                            except Exception as rename_err:
                                logger.error(f"Failed to rename file during migration {old_file_path} -> {new_file_path}: {rename_err}")
                        elif not old_file_path.exists() and new_sanitized_key != current_key:
                             logger.warning(f"Old file path '{old_file_path}' not found during migration, cannot rename.")


                        migrated_count += 1
                        needs_save_after_migration = True

                # Add except block for the inner try (catches errors during individual entry processing)
                except Exception as migration_entry_error:
                     logger.error(f"Error migrating entry for key '{current_key}': {migration_entry_error}")
                     # Continue to the next key even if one fails

            if migrated_count > 0:
                logger.info(f"Completed migration check. Migrated/updated {migrated_count} entries.")
            else:
                 logger.info("No metadata entries required migration.")

            # Save changes immediately if migration occurred
            if needs_save_after_migration:
                 logger.info("Saving migrated data structures to disk...")
                 self._save_to_disk()
            # --- End Migration Step ---


            # Save current provider info (do this *after* potential migration save)
            with open(embeddings_provider_file, 'w') as f:
                f.write("gemini")

            # Find .txt files whose *sanitized* names are not already loaded
            # Use the potentially updated self.chunks keys after migration
            existing_sanitized_names = set(self.chunks.keys())
            all_txt_files_on_disk = list(self.base_dir.glob('*.txt'))
            new_txt_files_to_load = []

            # Correct indentation for this block
            for txt_file_path in all_txt_files_on_disk:
                original_filename = txt_file_path.name
                sanitized_filename = self._sanitize_name(original_filename)
                if sanitized_filename not in existing_sanitized_names:
                    # Check if it's the internal list file (using sanitized name)
                    s_internal_list_name = self._sanitize_name(self._internal_list_doc_name)
                    if sanitized_filename == s_internal_list_name:
                        logger.info(f"Skipping internal list file found on disk: {original_filename}")
                        continue
                    new_txt_files_to_load.append(txt_file_path)
                else:
                    # If sanitized name exists, ensure metadata has original name
                    if 'original_name' not in self.metadata.get(sanitized_filename, {}):
                         logger.warning(f"Existing document '{sanitized_filename}' missing original name in metadata. Updating.")
                         self.metadata[sanitized_filename]['original_name'] = original_filename # Best guess

            if new_txt_files_to_load:
                logger.info(f"Found {len(new_txt_files_to_load)} new .txt files to load")
                for txt_file_path in new_txt_files_to_load:
                    original_filename = txt_file_path.name
                    try:
                        with open(txt_file_path, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                        # Pass the ORIGINAL filename to add_document
                        await self.add_document(original_filename, content, save_to_disk=False)
                        logger.info(f"Loaded and processed new file: {original_filename}")
                    except Exception as e:
                        logger.error(f"Error processing new file {original_filename}: {e}")
            else:
                logger.info("No new .txt files to load")

            # Update the internal document list after loading everything
            await self._update_document_list_file()

            # Save the updated state to disk once after all loading/updates
            # Only save here if no migration happened, otherwise it was saved earlier
            if not needs_save_after_migration:
                logger.info("Saving final document state after loading/checking.")
                self._save_to_disk()

        # Add the main except block for the entire _load_documents method
        except Exception as e:
            logger.error(f"Critical error during document loading/migration: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Reset state to prevent partial loads
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

    async def rename_document(self, old_name: str, new_name: str) -> str: # Make async
        """Rename a document in the system (regular doc, Google Doc, or lorebook)."""
        # Use original names for checks against internal list name
        if old_name == self._internal_list_doc_name:
            return f"Cannot rename the internal document list file '{old_name}'."
        s_internal_list_name = self._sanitize_name(self._internal_list_doc_name)
        s_new_name = self._sanitize_name(new_name)
        if s_new_name == s_internal_list_name:
             return f"Cannot rename a document to the internal list file name '{new_name}' (sanitized: {s_new_name})."

        result_message = f"Document '{old_name}' not found in the system" # Default message

        # --- Regular Document Rename ---
        s_old_name = self._get_sanitized_name_from_original(old_name)

        if s_old_name and s_old_name in self.metadata:
            logger.info(f"Attempting to rename regular document '{old_name}' (sanitized: {s_old_name}) to '{new_name}' (sanitized: {s_new_name})")

            # Check if new sanitized name conflicts with an existing document (excluding itself)
            if s_new_name in self.metadata and s_new_name != s_old_name:
                existing_original = self._get_original_name(s_new_name)
                return f"Cannot rename to '{new_name}': Sanitized name '{s_new_name}' conflicts with existing document '{existing_original}'."

            # Update the in-memory dictionaries using sanitized names
            self.chunks[s_new_name] = self.chunks.pop(s_old_name)
            if s_old_name in self.contextualized_chunks: # Handle contextualized chunks
                self.contextualized_chunks[s_new_name] = self.contextualized_chunks.pop(s_old_name)
            self.embeddings[s_new_name] = self.embeddings.pop(s_old_name)
            if s_old_name in self.bm25_indexes: # Handle BM25 index
                 self.bm25_indexes[s_new_name] = self.bm25_indexes.pop(s_old_name)

            # Update metadata: pop old, update new key, store original name
            meta = self.metadata.pop(s_old_name)
            meta['original_name'] = new_name # Update original name field
            meta['updated'] = datetime.now().isoformat()
            self.metadata[s_new_name] = meta

            # Save the changes to disk (pickles and json)
            self._save_to_disk()

            # Rename the actual file on disk using sanitized names
            old_file_path = self.base_dir / s_old_name
            if old_file_path.exists():
                new_file_path = self.base_dir / s_new_name
                try:
                    old_file_path.rename(new_file_path)
                    logger.info(f"Renamed file on disk from {s_old_name} to {s_new_name}")
                except Exception as e:
                    logger.error(f"Failed to rename file {old_file_path} to {new_file_path}: {e}")
                    # Consider how to handle this - maybe revert in-memory changes?

            result_message = f"Document renamed from '{old_name}' to '{new_name}'"
            await self._update_document_list_file() # Update list after rename
            return result_message

        # Check if it's a Google Doc
        tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
        if tracked_file.exists():
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
            
            # --- Google Doc Rename ---
            logger.info(f"Checking if '{old_name}' corresponds to a tracked Google Doc.")
            # Check if old_name is a Google Doc custom name or filename derived from ID
            doc_found = False
            for i, doc in enumerate(tracked_docs):
                doc_id = doc['id']
                current_custom_name = doc.get('custom_name')
                # Generate the potential filename based on ID (before sanitization)
                id_based_filename_orig = f"googledoc_{doc_id}.txt"

                # Check if old_name matches the current custom name OR the ID-based name
                if old_name == current_custom_name or old_name == id_based_filename_orig:
                    logger.info(f"Found matching Google Doc (ID: {doc_id}). Updating custom name to '{new_name}'.")
                    # Update the custom name in the tracking list
                    tracked_docs[i]['custom_name'] = new_name
                    doc_found = True

                    # Save the updated tracking list
                    with open(tracked_file, 'w') as f:
                        json.dump(tracked_docs, f, indent=2) # Add indent for readability

                    # Now, check if this Google Doc is *also* managed internally
                    # Determine the sanitized name it *would* have had based on its *previous* name
                    s_old_name_gdoc = self._get_sanitized_name_from_original(current_custom_name or id_based_filename_orig)

                    if s_old_name_gdoc and s_old_name_gdoc in self.metadata:
                        logger.info(f"Google Doc '{old_name}' is also managed internally (sanitized: {s_old_name_gdoc}). Renaming internal data.")
                        # Check for conflicts with the new sanitized name
                        if s_new_name in self.metadata and s_new_name != s_old_name_gdoc:
                             existing_original = self._get_original_name(s_new_name)
                             # Revert tracking file change before returning error? Maybe not necessary.
                             return f"Cannot rename Google Doc to '{new_name}': Sanitized name '{s_new_name}' conflicts with existing document '{existing_original}'."

                        # Rename internal data similar to regular documents
                        self.chunks[s_new_name] = self.chunks.pop(s_old_name_gdoc)
                        if s_old_name_gdoc in self.contextualized_chunks:
                            self.contextualized_chunks[s_new_name] = self.contextualized_chunks.pop(s_old_name_gdoc)
                        self.embeddings[s_new_name] = self.embeddings.pop(s_old_name_gdoc)
                        if s_old_name_gdoc in self.bm25_indexes:
                             self.bm25_indexes[s_new_name] = self.bm25_indexes.pop(s_old_name_gdoc)

                        meta = self.metadata.pop(s_old_name_gdoc)
                        meta['original_name'] = new_name # Update original name field
                        meta['updated'] = datetime.now().isoformat()
                        self.metadata[s_new_name] = meta

                        self._save_to_disk()

                        # Rename the file on disk using sanitized names
                        old_file_path = self.base_dir / s_old_name_gdoc
                        if old_file_path.exists():
                            new_file_path = self.base_dir / s_new_name
                            try:
                                old_file_path.rename(new_file_path)
                                logger.info(f"Renamed Google Doc file on disk from {s_old_name_gdoc} to {s_new_name}")
                            except Exception as e:
                                logger.error(f"Failed to rename Google Doc file {old_file_path} to {new_file_path}: {e}")
                    else:
                         logger.info(f"Google Doc '{old_name}' was tracked but not found in internal document manager storage.")


                    result_message = f"Google Doc renamed from '{old_name}' to '{new_name}' (tracking updated)"
                    await self._update_document_list_file() # Update list if internal data changed
                    return result_message # Exit loop once found and processed

            if doc_found: # Should have returned inside loop if found
                 pass # Should not reach here if found

        # --- Lorebook Rename ---
        # Lorebooks are not sanitized internally, handle directly by filename
        lorebooks_path = self.get_lorebooks_path()
        old_lorebook_path = lorebooks_path / old_name
        # Try adding .txt if the direct name doesn't exist
        if not old_lorebook_path.exists() and not old_name.endswith('.txt'):
            old_lorebook_path = lorebooks_path / f"{old_name}.txt"

        if old_lorebook_path.exists():
            logger.info(f"Attempting to rename lorebook: {old_lorebook_path}")
            # Ensure new name has .txt if old one did
            new_lorebook_name = new_name
            if old_lorebook_path.name.endswith('.txt') and not new_lorebook_name.endswith('.txt'):
                new_lorebook_name += '.txt'

            new_lorebook_path = lorebooks_path / new_lorebook_name

            # Check for conflict
            if new_lorebook_path.exists():
                 return f"Cannot rename lorebook to '{new_name}': File already exists."

            try:
                old_lorebook_path.rename(new_lorebook_path)
                result_message = f"Lorebook renamed from '{old_lorebook_path.name}' to '{new_lorebook_path.name}'"
                # Note: Lorebooks aren't in self.metadata, so list won't update automatically here.
                return result_message
            except Exception as e:
                 logger.error(f"Failed to rename lorebook {old_lorebook_path} to {new_lorebook_path}: {e}")
                 return f"Error renaming lorebook: {e}"

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
            result_message = f"Lorebook renamed from '{old_name}' to '{new_name}'"
            # Note: Lorebooks aren't in self.metadata, so list won't update automatically here.
            # This might be desired behavior, or _update_document_list_file needs adjustment
            # if lorebooks should also be listed. For now, assuming they aren't listed.
            return result_message

        return result_message # Return default "not found" if no match

    async def delete_document(self, name: str) -> bool: # Make async
        """Delete a document from the system using its original name."""
        # Prevent deletion of the internal list document
        if name == self._internal_list_doc_name:
            logger.warning(f"Attempted to delete the internal document list file '{name}'. Operation aborted.")
            return False

        deleted_something = False
        original_name_to_delete = name # Keep for logging/messages
        s_name_to_delete = self._get_sanitized_name_from_original(original_name_to_delete)

        try:
            # --- Regular Document Deletion ---
            if s_name_to_delete and s_name_to_delete in self.metadata:
                logger.info(f"Attempting to delete regular document '{original_name_to_delete}' (sanitized: {s_name_to_delete})")
                # Remove from memory
                if s_name_to_delete in self.chunks: del self.chunks[s_name_to_delete]
                if s_name_to_delete in self.contextualized_chunks: del self.contextualized_chunks[s_name_to_delete]
                if s_name_to_delete in self.embeddings: del self.embeddings[s_name_to_delete]
                if s_name_to_delete in self.bm25_indexes: del self.bm25_indexes[s_name_to_delete]
                del self.metadata[s_name_to_delete] # Delete metadata last

                # Save changes to pickles/json
                self._save_to_disk()

                # Remove file from disk using sanitized name
                file_path = self.base_dir / s_name_to_delete
                if file_path.exists():
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted file from disk: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {e}")
                deleted_something = True

            # --- Lorebook Deletion ---
            # Lorebooks use original names for files
            lorebooks_path = self.get_lorebooks_path()
            lorebook_path = lorebooks_path / original_name_to_delete
            # Try adding .txt if direct name doesn't exist
            if not lorebook_path.exists() and not original_name_to_delete.endswith('.txt'):
                lorebook_path = lorebooks_path / f"{original_name_to_delete}.txt"

            if lorebook_path.exists():
                logger.info(f"Attempting to delete lorebook file: {lorebook_path}")
                try:
                    lorebook_path.unlink()
                    logger.info(f"Deleted lorebook file: {lorebook_path}")
                    deleted_something = True
                except Exception as e:
                    logger.error(f"Failed to delete lorebook file {lorebook_path}: {e}")

            # --- Google Doc Tracking Deletion ---
            tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
            if tracked_file.exists():
                try:
                    with open(tracked_file, 'r') as f:
                        tracked_docs = json.load(f)

                    original_length = len(tracked_docs)
                    # Find the doc to remove based on custom name or ID-based filename matching the *original* name provided
                    docs_to_keep = []
                    removed_gdoc = False
                    for doc in tracked_docs:
                        doc_id = doc['id']
                        custom_name = doc.get('custom_name')
                        id_based_filename_orig = f"googledoc_{doc_id}.txt"

                        # Check if the original name matches either the custom name or the ID-based name
                        if not (custom_name == original_name_to_delete or id_based_filename_orig == original_name_to_delete):
                            docs_to_keep.append(doc)
                        else:
                            logger.info(f"Found tracked Google Doc entry corresponding to '{original_name_to_delete}' (ID: {doc_id}). Removing from tracking.")
                            removed_gdoc = True

                    if removed_gdoc:
                        with open(tracked_file, 'w') as f:
                            json.dump(docs_to_keep, f, indent=2)
                        deleted_something = True # Mark as deleted if tracking entry was removed

                except Exception as track_e:
                    logger.error(f"Error updating tracked Google Docs file while deleting '{original_name_to_delete}': {track_e}")
                    # Don't fail the whole delete operation, just log the tracking error

            # --- Final Steps ---
            if deleted_something:
                 logger.info(f"Successfully completed deletion operations for '{original_name_to_delete}'")
                 # Update the list file only if a *managed* document was deleted (i.e., s_name_to_delete was valid)
                 if s_name_to_delete:
                     await self._update_document_list_file()
                 return True
            else:
                 logger.warning(f"Document, lorebook, or Google Doc tracking entry '{original_name_to_delete}' not found for deletion.")
                 return False # Return False if nothing was found/deleted

        except Exception as e:
            logger.error(f"Error during deletion process for '{original_name_to_delete}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        except Exception as e:
            logger.error(f"Error deleting document {name}: {e}")
            return False


    async def rerank_results(self, query: str, initial_results: List[Tuple], top_k: int = None) -> List[Tuple]: # Make async
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
        
        # 1. Generate a query-focused embedding that represents what we're looking for asynchronously
        query_context = f"Question: {query}\nWhat information would fully answer this question?"
        query_embedding_result = await self.generate_embeddings([query_context], is_query=True) # Await async call
        if query_embedding_result.size == 0:
            logger.error("Failed to generate query embedding for reranking.")
            return initial_results # Return original results if embedding fails
        query_embedding = query_embedding_result[0]


        # 2. Generate content-focused embeddings for each chunk asynchronously
        content_texts = [f"This document contains the following information: {chunk}" for chunk in chunks]
        content_embeddings = await self.generate_embeddings(content_texts, is_query=False) # Await async call
        if content_embeddings.size == 0 or content_embeddings.shape[0] != len(content_texts):
             logger.error("Failed to generate content embeddings for reranking or mismatch in count.")
             return initial_results # Return original results if embedding fails


        # 3. Calculate relevance scores using these specialized embeddings
        # Ensure embeddings are compatible for dot product
        if query_embedding.shape[0] != content_embeddings.shape[1]:
             logger.error(f"Embedding dimension mismatch for reranking: Query({query_embedding.shape[0]}) vs Content({content_embeddings.shape[1]})")
             return initial_results

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

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
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from textwrap import shorten
from rank_bm25 import BM25Okapi

from utils.logging import sanitize_for_logging

logger = logging.getLogger(__name__)

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
            # Get current document names, excluding the list file itself
            current_doc_names = sorted([
                name for name in self.metadata.keys()
                if name != self._internal_list_doc_name
            ])

            # Format the content
            if not current_doc_names:
                content = "No documents are currently managed."
            else:
                content = "Managed Documents:\n\n" + "\n".join(f"- {name}" for name in current_doc_names)

            # Check if the document exists and if content needs updating
            needs_update = True
            list_doc_path = self.base_dir / self._internal_list_doc_name
            if list_doc_path.exists():
                try:
                    with open(list_doc_path, 'r', encoding='utf-8-sig') as f:
                        existing_content = f.read()
                    if existing_content == content:
                        needs_update = False
                        logger.info("Internal document list content is already up-to-date.")
                except Exception as e:
                    logger.warning(f"Could not read existing internal document list file: {e}. Will overwrite.")

            if needs_update:
                logger.info(f"Content for {self._internal_list_doc_name} requires update. Adding/updating document.")
                # Use add_document to ensure it's processed like other docs (chunked, embedded)
                # Pass _internal_call=True to prevent recursion
                success = await self.add_document(self._internal_list_doc_name, content, save_to_disk=True, _internal_call=True)
                if success:
                    logger.info(f"Successfully updated and saved {self._internal_list_doc_name}.")
                else:
                    logger.error(f"Failed to add/update {self._internal_list_doc_name} due to errors during processing or saving.")
            else:
                 # Even if content didn't change, ensure metadata reflects check time
                 if self._internal_list_doc_name in self.metadata:
                     self.metadata[self._internal_list_doc_name]['checked'] = datetime.now().isoformat()
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
            #if len([k for k in combined_scores.keys() if k == key]) <= 3:
            #    logger.info(f"Top BM25 result: {doc_name}, score: {score:.4f}, normalized: {norm_score:.4f}")
        
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
            # Prevent direct modification of the internal list doc via this method if not internal call
            if name == self._internal_list_doc_name and not _internal_call:
                 logger.warning(f"Attempted to directly modify internal document '{name}'. Use specific commands or let the system manage it.")
                 # Optionally, raise an error or return False
                 # raise ValueError(f"Cannot directly modify internal document '{name}'")
                 return False # Indicate failure

            if not content or not content.strip():
                logger.warning(f"Document {name} has no content. Skipping.")
                return True # Technically not a failure, just skipped

            # Calculate word count
            word_count = len(content.split())
            max_words_for_context = 30000 # Define the limit
            logger.info(f"Document '{name}' word count: {word_count}")

            # Create original chunks (always needed)
            chunks = self._chunk_text(content)
            if not chunks:
                logger.warning(f"Document {name} has no content to chunk. Skipping.")
                return True # Technically not a failure, just skipped

            # Check if document already exists and if content has changed
            content_changed = True
            current_hash = None # Initialize hash variable
            if name in self.metadata and 'content_hash' in self.metadata[name]:
                import hashlib
                current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                previous_hash = self.metadata[name]['content_hash']
                if previous_hash and previous_hash == current_hash:
                    content_changed = False
                    logger.info(f"Document {name} content has not changed based on hash.")

            # Determine if processing (embedding, indexing) is needed
            # Needs processing if content changed OR if embeddings are missing for this doc
            needs_processing = content_changed or name not in self.embeddings

            if needs_processing:
                logger.info(f"Processing required for document {name} (Content changed: {content_changed}, Embeddings missing: {name not in self.embeddings})")

                # Decide whether to contextualize based on word count
                should_contextualize = word_count <= max_words_for_context

                if should_contextualize:
                    logger.info(f"Document {name} ({word_count} words) is within limit ({max_words_for_context}). Generating contextualized chunks.")
                    contextualized_chunks = await self.contextualize_chunks(name, content, chunks)
                    # Use contextualized chunks for embeddings and BM25
                    chunks_for_embedding = contextualized_chunks
                    chunks_for_bm25 = contextualized_chunks
                    # Store the contextualized chunks
                    self.contextualized_chunks[name] = contextualized_chunks
                else:
                    logger.warning(f"Document {name} ({word_count} words) exceeds limit ({max_words_for_context}). Skipping contextualization, using original chunks.")
                    # Use original chunks for embeddings and BM25
                    chunks_for_embedding = chunks
                    chunks_for_bm25 = chunks
                    # Ensure no stale contextualized chunks exist if the doc previously fit
                    if name in self.contextualized_chunks:
                        del self.contextualized_chunks[name]
                        logger.info(f"Removed previous contextualized chunks for oversized document {name}.")

                # Generate embeddings using the selected chunks (either original or contextualized)
                # Ensure chunks_for_embedding is not empty before proceeding
                if not chunks_for_embedding:
                    logger.error(f"No chunks available for embedding document {name}. Skipping embedding generation.")
                    embeddings = np.array([]) # Assign empty array if no chunks
                else:
                    titles = [name] * len(chunks_for_embedding) # Title count should match chunk count
                    embeddings = await self.generate_embeddings(chunks_for_embedding, is_query=False, titles=titles)

                # Store document data
                self.chunks[name] = chunks  # Always store original chunks
                self.embeddings[name] = embeddings # Store the generated embeddings

                # Calculate hash if not already done (e.g., if content changed but wasn't in metadata before)
                if current_hash is None:
                     import hashlib
                     current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                # Update metadata
                self.metadata[name] = {
                    'added': self.metadata.get(name, {}).get('added', datetime.now().isoformat()), # Preserve original add time
                    'updated': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'content_hash': current_hash,
                    'contextualized': should_contextualize # Add flag indicating if contextualized
                }

                # Create BM25 index using the selected chunks
                # Ensure chunks_for_bm25 is not empty
                if chunks_for_bm25:
                    self.bm25_indexes[name] = self._create_bm25_index(chunks_for_bm25)
                else:
                    logger.warning(f"No chunks available for BM25 indexing document {name}. Skipping BM25 index creation.")
                    if name in self.bm25_indexes: # Remove stale index if it exists
                        del self.bm25_indexes[name]


                logger.info(f"Finished processing for document {name}. Contextualized: {should_contextualize}")

            else: # Content hasn't changed, just update timestamp
                logger.info(f"Document {name} content unchanged. Updating 'checked' timestamp.")
                if name in self.metadata:
                    self.metadata[name]['checked'] = datetime.now().isoformat()
                else: # Should not happen if content_changed is False, but safety check
                     self.metadata[name] = {
                        'checked': datetime.now().isoformat(),
                        'chunk_count': len(chunks) # Best guess
                     }

            # Save to disk if requested
            if save_to_disk:
                self._save_to_disk() # This will raise exceptions if it fails

            # Log completion message
            if needs_processing:
                 log_context_status = self.metadata.get(name, {}).get('contextualized', 'N/A')
                 logger.info(f"{'Internally added/updated' if _internal_call else 'Added/updated'} document: {name} with {len(chunks)} chunks. Contextualized: {log_context_status}")
            else:
                 logger.info(f"Document {name} verified unchanged")

            # Update the document list file, unless this was an internal call
            if not _internal_call:
                await self._update_document_list_file()

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
            #logger.info(f"Using {'contextualized' if is_contextualized else 'original'} chunk")
            
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

            # Iterate through all documents
            for doc_name, chunks in self.chunks.items():
                logger.info(f"Regenerating embeddings for document: {doc_name}")

                # Get contextualized chunks (regenerate if needed, although ideally they exist)
                if doc_name not in self.contextualized_chunks:
                     logger.warning(f"Contextualized chunks missing for {doc_name}, regenerating...")
                     doc_path = self.base_dir / doc_name
                     if doc_path.exists():
                         with open(doc_path, 'r', encoding='utf-8-sig') as f:
                             doc_content = f.read()
                         self.contextualized_chunks[doc_name] = await self.contextualize_chunks(doc_name, doc_content, chunks)
                     else:
                         logger.error(f"Cannot find original file {doc_name} to regenerate contextualized chunks.")
                         continue # Skip this document if original file is missing

                contextualized_chunks_for_doc = self.contextualized_chunks.get(doc_name, chunks) # Fallback to original if still missing

                # Generate new embeddings asynchronously
                titles = [doc_name] * len(contextualized_chunks_for_doc)
                new_embeddings = await self.generate_embeddings(contextualized_chunks_for_doc, is_query=False, titles=titles) # Await async call

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

                        # Generate embeddings asynchronously
                        titles = [doc_name] * len(contextualized_chunks)
                        self.embeddings[doc_name] = await self.generate_embeddings(contextualized_chunks, is_query=False, titles=titles) # Await async call

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
                        # Use save_to_disk=False here to avoid saving after each file
                        await self.add_document(txt_file.name, content, save_to_disk=False)
                        logger.info(f"Loaded and processed {txt_file.name}")
                    except Exception as e:
                        logger.error(f"Error processing {txt_file.name}: {e}")
            else:
                logger.info("No new .txt files to load")

            # Update the internal document list after loading everything
            await self._update_document_list_file()

            # Save the updated state to disk once after all loading/updates
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

    async def rename_document(self, old_name: str, new_name: str) -> str: # Make async
        """Rename a document in the system (regular doc, Google Doc, or lorebook)."""
        # Prevent renaming the internal list document
        if old_name == self._internal_list_doc_name:
            return f"Cannot rename the internal document list file '{old_name}'."
        if new_name == self._internal_list_doc_name:
             return f"Cannot rename a document to the internal list file name '{new_name}'."

        result_message = f"Document '{old_name}' not found in the system" # Default message

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

            result_message = f"Document renamed from '{old_name}' to '{new_name}'"
            await self._update_document_list_file() # Update list after rename
            return result_message

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

                    result_message = f"Google Doc renamed from '{old_name}' to '{new_name}'"
                    await self._update_document_list_file() # Update list after rename
                    return result_message

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
        """Delete a document from the system."""
        # Prevent deletion of the internal list document
        if name == self._internal_list_doc_name:
            logger.warning(f"Attempted to delete the internal document list file '{name}'. Operation aborted.")
            return False

        deleted = False
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
                deleted = True

            # Check if it's a lorebook
            lorebooks_path = self.get_lorebooks_path()
            lorebook_path = lorebooks_path / name
            
            # Try with .txt extension if not found
            if not lorebook_path.exists() and not name.endswith('.txt'):
                lorebook_path = lorebooks_path / f"{name}.txt"
            if lorebook_path.exists():
                lorebook_path.unlink()
                # Lorebooks aren't in self.metadata, so list won't update automatically.
                deleted = True # Mark as deleted, but list won't change unless logic is added

            # Check if it's a tracked Google Doc and remove from tracking if so
            tracked_file = Path(self.base_dir) / "tracked_google_docs.json"
            if tracked_file.exists():
                try:
                    with open(tracked_file, 'r') as f:
                        tracked_docs = json.load(f)
                    
                    # Find the doc to remove based on filename or custom name matching 'name'
                    original_length = len(tracked_docs)
                    tracked_docs = [
                        doc for doc in tracked_docs 
                        if not (
                            (doc.get('custom_name') == name) or 
                            (f"googledoc_{doc['id']}.txt" == name) or
                            (doc.get('custom_name') == name.replace('.txt', '')) or # Handle cases where .txt might be missing
                            (f"googledoc_{doc['id']}" == name.replace('.txt', ''))
                        )
                    ]
                    
                    if len(tracked_docs) < original_length:
                        logger.info(f"Removed tracked Google Doc entry corresponding to '{name}'")
                        with open(tracked_file, 'w') as f:
                            json.dump(tracked_docs, f)
                        # If we removed a tracked doc, mark as deleted
                        deleted = True

                except Exception as track_e:
                    logger.error(f"Error updating tracked Google Docs file while deleting {name}: {track_e}")
                    # Don't fail the whole delete operation, just log the tracking error

            if deleted:
                 logger.info(f"Successfully deleted document/lorebook/tracking entry for '{name}'")
                 # Update the list file only if a *managed* document was deleted
                 if name in self.metadata: # Check if it was in metadata before deletion
                     await self._update_document_list_file()
                 return True
            else:
                 logger.warning(f"Document '{name}' not found for deletion.")
                 return False # Return False if not found

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

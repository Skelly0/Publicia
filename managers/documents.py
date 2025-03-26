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
        
        # Initialize contextualization stats 
        # (actual values will be loaded from disk when needed)
        self._contextualization_count = 0
        self._contextualization_cost = 0.0
        
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

    def _estimate_batch_cost(self, batch, document_content, model):
        """Estimate cost based on document size, batch size, and specific model."""
        # System prompt tokens (~150)
        system_tokens = 150
        
        # Document content tokens (first 2000 chars)
        doc_tokens = min(len(document_content[:2000].split()) * 1.3, 500)
        
        # Instructions/formatting tokens (~100 base + 25 per chunk)
        instruction_tokens = 100 + (25 * len(batch))
        
        # Chunk content tokens (each chunk limited to 300 chars)
        chunk_tokens = sum(len(chunk[:300].split()) * 1.3 for chunk in batch)
        
        # Expected output tokens (~40 per chunk)
        output_tokens = 40 * len(batch)
        
        # Total estimate
        total_tokens = system_tokens + doc_tokens + instruction_tokens + chunk_tokens + output_tokens
        
        # Get model-specific cost
        cost_per_1k = self.config.MODEL_COSTS.get(model, self.config.MODEL_COSTS["default"])
        
        return (total_tokens / 1000) * cost_per_1k
    
    def _get_actual_cost(self, tokens_used, model):
        """Calculate actual cost based on tokens used and model."""
        cost_per_1k = self.config.MODEL_COSTS.get(model, self.config.MODEL_COSTS["default"])
        return (tokens_used / 1000) * cost_per_1k
    
    def _load_contextualization_stats(self):
        """Load usage statistics with daily auto-reset."""
        # Removed the hasattr check to ensure stats are always loaded from file on init
        
        stats_path = Path(self.base_dir) / 'contextualization_stats.json'
        
        try:
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    
                # Check if we need to reset for a new day
                last_reset = datetime.fromisoformat(stats.get('last_reset', '2000-01-01'))
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                if last_reset < today:
                    # New day - reset counters
                    logger.info("New day detected, resetting contextualization counters")
                    self._contextualization_count = 0
                    self._contextualization_cost = 0.0
                    self._save_contextualization_stats()
                else:
                    # Same day - load current counters
                    self._contextualization_count = stats.get('count', 0)
                    self._contextualization_cost = stats.get('cost', 0.0)
            else:
                # First run - initialize counters
                self._contextualization_count = 0
                self._contextualization_cost = 0.0
                self._save_contextualization_stats()
        except Exception as e:
            logger.error(f"Error loading contextualization stats: {e}")
            self._contextualization_count = 0
            self._contextualization_cost = 0.0

    def _save_contextualization_stats(self):
        """Save usage statistics to disk."""
        stats_path = Path(self.base_dir) / 'contextualization_stats.json'
        try:
            stats = {
                'count': self._contextualization_count,
                'cost': self._contextualization_cost,
                'last_reset': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            logger.error(f"Error saving contextualization stats: {e}")
    
    async def _generate_batch_contexts(self, document_content: str, chunks: List[str]) -> Tuple[List[str], str, int]:
        """Generate contexts for multiple chunks in one API call, with model fallbacks."""
        # Create batch prompt
        system_prompt = """You are a helpful AI assistant that creates concise contextual descriptions for document chunks. 
Your task is to provide short, succinct contexts that situate specific chunks within the overall document.
Output ONLY the contexts, one per line, in the same order as the chunks. No explanations or additional text."""
        
        user_prompt = f"""<document>\n{document_content[:2000]}\n</document>\n
I need contextual descriptions for the following {len(chunks)} chunks from this document.
Provide ONLY one line per chunk with a brief context that helps situate it within the document.
Do not number your responses. Just provide one context per line."""
        
        # Add each chunk to the prompt
        for i, chunk in enumerate(chunks):
            user_prompt += f"\nCHUNK {i+1}:\n{chunk[:300]}\n"
        
        # Prioritize models by cost (free first, then cheapest to most expensive)
        fallback_models = [
            # Free models first
            #"google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-exp:free", 
            "google/gemma-3-27b-it:free",
            
            # Then in order of increasing cost
            "google/gemini-2.0-flash-lite-001",   # $0.075
            "cohere/command-r7b-12-2024",         # $0.0375
            "microsoft/phi-4",                    # $0.05
            "google/gemini-2.0-flash-001"         # $0.1
        ]
        
        # API headers
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://discord.com", 
            "X-Title": "Publicia - Batch Context",
            "Content-Type": "application/json"
        }
        
        # Try each model in sequence until one works
        for model in fallback_models:
            # Check if we can afford this model
            estimated_cost = self._estimate_batch_cost(chunks, document_content, model)
            remaining_budget = self.config.CONTEXTUALIZATION_MAX_DAILY_BUDGET - self._contextualization_cost
            
            # Skip paid models that would exceed budget
            if estimated_cost > remaining_budget and self.config.MODEL_COSTS.get(model, 0) > 0:
                logger.warning(f"Model {model} would exceed budget (est. ${estimated_cost:.2f}, remaining: ${remaining_budget:.2f})")
                continue
            
            logger.info(f"Attempting batch context with model: {model} for {len(chunks)} chunks")
            
            # Prepare API payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 50 * len(chunks)  # ~50 tokens per context
            }
            
            # API call and response processing
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120  # Longer timeout for batches
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API error with model {model}: {error_text}")
                            continue
                            
                        completion = await response.json()
                
                # Process response
                if completion and completion.get('choices') and len(completion['choices']) > 0:
                    content = completion['choices'][0]['message']['content'].strip()
                    
                    # Split into lines to get individual contexts
                    contexts = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Get token usage for accurate costing
                    tokens_used = completion.get('usage', {}).get('total_tokens', 0)
                    
                    # Estimate token count if not provided by API
                    if not tokens_used:
                        tokens_used = len(system_prompt.split()) + len(user_prompt.split()) + len(content.split())
                        tokens_used = int(tokens_used * 1.3)  # Add 30% for tokenization
                    
                    # Handle mismatched context count
                    if len(contexts) == len(chunks):
                        logger.info(f"Successfully generated {len(contexts)} contexts with model {model}")
                        return contexts, model, tokens_used
                    
                    logger.warning(f"Generated {len(contexts)} contexts but expected {len(chunks)}")
                    
                    # Try to salvage by padding or truncating
                    if len(contexts) > len(chunks):
                        return contexts[:len(chunks)], model, tokens_used
                    else:
                        return contexts + [None] * (len(chunks) - len(contexts)), model, tokens_used
                
            except Exception as e:
                logger.error(f"Error with model {model}: {e}")
                continue
        
        # If all models failed, return empty contexts
        logger.error("All models failed to generate batch contexts")
        return [None] * len(chunks), "none", 0
    
    async def contextualize_chunks(self, doc_name: str, document_content: str, chunks: List[str]) -> List[str]:
        """Generate context for chunks with batch processing and budget controls."""
        # Skip if disabled
        if not self.config.CONTEXTUALIZATION_ENABLED:
            logger.info(f"Contextualization disabled, skipping for document: {doc_name}")
            return chunks
        
        # Load stats
        self._load_contextualization_stats()
        
        # Check daily limits
        if self._contextualization_count >= self.config.CONTEXTUALIZATION_MAX_DAILY:
            logger.warning(f"Daily chunk limit reached ({self._contextualization_count}/{self.config.CONTEXTUALIZATION_MAX_DAILY})")
            return chunks
        
        if self._contextualization_cost >= self.config.CONTEXTUALIZATION_MAX_DAILY_BUDGET:
            logger.warning(f"Daily budget limit reached (${self._contextualization_cost:.2f}/${self.config.CONTEXTUALIZATION_MAX_DAILY_BUDGET:.2f})")
            return chunks
        
        logger.info(f"Starting contextualization for {doc_name} with {len(chunks)} chunks")
        
        # Start with copies of original chunks
        contextualized_chunks = chunks.copy()
        
        # Process in batches
        batch_size = self.config.CONTEXTUALIZATION_BATCH_SIZE
        for i in range(0, len(chunks), batch_size):
            # Get current batch
            batch = chunks[i:i+batch_size]
            
            # Check remaining chunk limit
            remaining_limit = self.config.CONTEXTUALIZATION_MAX_DAILY - self._contextualization_count
            if len(batch) > remaining_limit:
                logger.warning(f"Approaching daily limit, processing only {remaining_limit} chunks")
                batch = batch[:remaining_limit]
                
            if not batch:
                continue
            
            # Generate contexts for batch
            contexts, model_used, tokens_used = await self._generate_batch_contexts(document_content, batch)
            
            # If generation successful
            if contexts and model_used != "none":
                # Calculate actual cost based on model used
                actual_cost = self._get_actual_cost(tokens_used, model_used)
                
                # Update running totals
                self._contextualization_count += len(batch)
                self._contextualization_cost += actual_cost
                self._save_contextualization_stats()
                
                # Apply contexts to chunks
                for j, context in enumerate(contexts):
                    if context and i + j < len(contextualized_chunks):
                        contextualized_chunks[i + j] = f"{context} {chunks[i + j]}"
                
                # Log progress
                logger.info(f"Batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} " +
                           f"using {model_used} cost: ${actual_cost:.4f}, running: ${self._contextualization_cost:.2f}")
            
            # Stop if limits reached
            if self._contextualization_count >= self.config.CONTEXTUALIZATION_MAX_DAILY:
                logger.warning("Daily chunk limit reached during processing, stopping")
                break
                
            if self._contextualization_cost >= self.config.CONTEXTUALIZATION_MAX_DAILY_BUDGET:
                logger.warning("Daily budget limit reached during processing, stopping")
                break
        
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
            
            # Check if document already exists and if content has changed
            content_changed = True
            if name in self.chunks:
                # Calculate hash of current content
                import hashlib
                current_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                
                # Check if we have a stored hash
                previous_hash = None
                if name in self.metadata and 'content_hash' in self.metadata[name]:
                    previous_hash = self.metadata[name]['content_hash']
                
                # Compare hashes to determine if content has changed
                if previous_hash and previous_hash == current_hash:
                    content_changed = False
                    logger.info(f"Document {name} content has not changed, skipping contextualized chunks regeneration")
            
            # Generate contextualized chunks only if content has changed or document is new
            if content_changed or name not in self.contextualized_chunks:
                logger.info(f"Generating contextualized chunks for document {name}")
                contextualized_chunks = await self.contextualize_chunks(name, content, chunks)
                
                # Generate embeddings using contextualized chunks
                # Use document name as title for all chunks to improve embedding quality
                titles = [name] * len(contextualized_chunks)
                embeddings = self.generate_embeddings(contextualized_chunks, is_query=False, titles=titles)
                
                # Store document data
                self.chunks[name] = chunks  # Store original chunks
                self.contextualized_chunks[name] = contextualized_chunks  # Store contextualized chunks
                self.embeddings[name] = embeddings
                
                # Calculate and store content hash
                import hashlib
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                
                # Update metadata
                self.metadata[name] = {
                    'added': datetime.now().isoformat(),
                    'updated': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'content_hash': content_hash
                }
                
                # Create BM25 index for contextualized chunks
                self.bm25_indexes[name] = self._create_bm25_index(contextualized_chunks)
            else:
                # Update only the timestamp in metadata
                self.metadata[name]['checked'] = datetime.now().isoformat()
            
            # Save to disk only if requested
            if save_to_disk:
                self._save_to_disk()
            
            if content_changed or name not in self.chunks:
                logger.info(f"Added/updated document: {name} with {len(chunks)} chunks and contextualized embeddings")
            else:
                logger.info(f"Document {name} verified unchanged")
            
        except Exception as e:
            logger.error(f"Error adding document {name}: {e}")
            raise

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
        """Rename a document in the system (regular doc, Google Doc, or lorebook)."""
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
                        # If we removed a tracked doc, we can assume success even if it wasn't in chunks/metadata
                        return True 
                            
                except Exception as track_e:
                    logger.error(f"Error updating tracked Google Docs file while deleting {name}: {track_e}")
                    # Don't fail the whole delete operation, just log the tracking error

            return False # Return False if not found in regular docs, lorebooks, or tracked docs
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

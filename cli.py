#!/usr/bin/env python3
"""
Publicia CLI - Command Line Interface for Querying
Allows asking questions to the Publicia knowledge base directly from the terminal.
"""
import argparse
import asyncio
import sys
import os
import json
import base64
import logging
import re
import io
import aiohttp
import numpy as np
from textwrap import shorten
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Add project root to sys.path to allow imports from managers, utils etc.
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logging import configure_logging # Use existing logging setup
from managers.config import Config
from managers.documents import DocumentManager
from managers.images import ImageManager
from managers.preferences import UserPreferencesManager # Keep for potential future use (e.g., model preference)
from prompts.system_prompt import SYSTEM_PROMPT
from utils.helpers import is_image # Re-use helper

# Configure logging for the CLI
logger = configure_logging() # Use the existing setup

# --- Adapted Helper Functions from bot.py ---

async def _download_image_to_base64(image_url: str) -> Optional[str]:
    """Download an image from a URL and convert it to base64."""
    try:
        logger.info(f"Attempting to download image from URL: {image_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to download image: {resp.status} {resp.reason}")
                    return None

                image_data = await resp.read()
                mime_type = resp.headers.get('Content-Type', 'image/jpeg') # Default to jpeg
                if not mime_type.startswith('image/'):
                    logger.error(f"URL content type is not an image: {mime_type}")
                    return None

                # Convert to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"Successfully downloaded and encoded image ({len(image_data)} bytes, type: {mime_type})")
                return f"data:{mime_type};base64,{base64_data}"
    except aiohttp.ClientError as e:
        logger.error(f"Network error downloading image: {e}")
        return None
    except Exception as e:
        logger.error(f"Error downloading or processing image: {e}")
        return None

async def _cli_try_ai_completion(
    config: Config,
    initial_model: str, # Renamed parameter
    messages: List[Dict],
    image_attachments: Optional[List[str]] = None,
    temperature: float = 0.1,
    timeout_duration: int = 300
) -> Tuple[Optional[Dict], Optional[str]]: # Return completion and model used
    """AI completion call for CLI use with fallback."""

    # Define fallback models
    fallback_models = [
        "qwen/qwq-32b",
        "google/gemini-2.5-flash-preview",
        "mistralai/mistral-large"
        # Add more paid/reliable models here if needed
    ]

    models_to_try = [initial_model] + [m for m in fallback_models if m != initial_model] # Ensure initial model is first and no duplicates

    logger.info(f"Attempting completion with models: {', '.join(models_to_try)}")

    # Vision capable models list (copied from bot.py for now)
    # TODO: Consider centralizing this list
    vision_capable_models = [
        "google/gemini-2.5-flash-preview", "google/gemini-2.0-pro-exp-02-05",
        "google/gemini-2.5-pro-exp-03-25", "microsoft/phi-4-multimodal-instruct",
        "anthropic/claude-3.7-sonnet:beta", "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet:beta", "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku:beta", "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-haiku:beta"
    ]

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost/cli", # Referer for CLI
        "X-Title": "Publicia - CLI",
        "Content-Type": "application/json"
    }

    for current_model in models_to_try:
        logger.info(f"Trying model: {current_model}")

        processed_messages = messages.copy()
        need_vision = image_attachments and len(image_attachments) > 0
        is_vision_model = current_model in vision_capable_models

        if need_vision:
            if not is_vision_model:
                logger.warning(f"Model {current_model} does not support vision, but images were provided. Ignoring images.")
            else:
                # Add images to the last user message
                for i in range(len(processed_messages) - 1, -1, -1):
                            if processed_messages[i]["role"] == "user":
                                user_msg = processed_messages[i]
                                text_content = user_msg["content"]
                                content_array = [{"type": "text", "text": text_content}]
                                for img_data in image_attachments:
                                    if img_data:
                                        content_array.append({
                                            "type": "image_url",
                                            "image_url": {"url": img_data}
                                        })
                                processed_messages[i]["content"] = content_array
                                logger.info(f"Added {len(content_array) - 1} images to message for vision model {current_model}")
                                break

        payload = {
            "model": current_model, # Use current_model in loop
            "messages": processed_messages,
            "temperature": temperature,
            "max_tokens": 8000 # Keep max tokens high
        }

        # Add provider config if available
        provider_config = config.get_provider_config(current_model)
        if provider_config:
            payload["provider"] = provider_config
            logger.info(f"Using custom provider configuration for {current_model}: {provider_config}")

        # Log sanitized messages
        sanitized_messages = []
        for msg in processed_messages:
            content_repr = ""
            if isinstance(msg["content"], list):
                image_count = sum(1 for item in msg["content"] if item.get("type") == "image_url")
                text_parts = [item["text"] for item in msg["content"] if item.get("type") == "text"]
                text_content = " ".join(text_parts)
                content_repr = f"{shorten(text_content, width=100, placeholder='...')} [+ {image_count} images]"
            else:
                content_repr = shorten(msg["content"], width=100, placeholder='...')
            sanitized_messages.append({"role": msg["role"], "content": content_repr})
        logger.debug(f"Request payload messages for {current_model}: {json.dumps(sanitized_messages, indent=2)}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout_duration
                ) as response:
                    if response.status == 429: # Specifically handle rate limit
                        logger.warning(f"Rate limit hit for model {current_model}. Trying next model.")
                        continue # Move to the next model in the list

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error (Status {response.status}) for model {current_model}: {error_text}")
                        continue # Try next model on other errors too

                    completion = await response.json()

                    # Validate the response structure
                    if completion and completion.get('choices') and len(completion['choices']) > 0:
                        if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                            logger.info(f"Successfully received valid completion structure from {current_model}")
                            # Return completion AND the model that succeeded
                            return completion, current_model
                        else:
                            logger.error(f"LLM response structure invalid (missing message/content) from {current_model}: {completion}")
                            # Don't immediately fail, try next model
                    else:
                        logger.error(f"LLM response structure invalid (missing choices) from {current_model}: {completion}")
                        # Don't immediately fail, try next model

        except asyncio.TimeoutError:
            logger.error(f"API call timed out after {timeout_duration}s for model {current_model}. Trying next model.")
        except aiohttp.ClientError as e:
             logger.error(f"Network error during API call for model {current_model}: {e}. Trying next model.")
        except Exception as e:
            logger.error(f"Unexpected error during API call for model {current_model}: {e}. Trying next model.", exc_info=True)

        # If we reach here, the current model failed, loop continues to the next

    logger.error(f"All models failed to generate a valid completion. Tried: {', '.join(models_to_try)}")
    return None, None # Return None for both if all fail

# --- Core CLI Logic ---

async def initialize_managers():
    """Initialize and load necessary managers."""
    logger.info("Initializing managers...")
    try:
        config = Config()
        doc_manager = DocumentManager(config=config)
        image_manager = ImageManager(document_manager=doc_manager) # Needs doc_manager
        pref_manager = UserPreferencesManager() # Keep for consistency, might use later

        logger.info("Loading documents...")
        await doc_manager._load_documents() # Ensure documents are loaded
        # Image metadata is loaded automatically in ImageManager.__init__
        logger.info("Image manager initialized (metadata loaded).")

        logger.info("Managers initialized successfully.")
        return config, doc_manager, image_manager, pref_manager
    except Exception as e:
        logger.critical(f"Failed to initialize managers: {e}", exc_info=True)
        sys.exit(1) # Exit if managers fail to load

async def process_cli_query(args: argparse.Namespace, config: Config, doc_manager: DocumentManager, image_manager: ImageManager, pref_manager: UserPreferencesManager):
    """Handles the query processing for the CLI."""
    question = args.question
    image_url = args.image_url
    preferred_model = args.model or config.LLM_MODEL # Use CLI arg or default

    # Define vision models list here for use in this function
    # TODO: Consider centralizing this list
    vision_capable_models = [
        "google/gemini-2.5-flash-preview", "google/gemini-2.0-pro-exp-02-05",
        "google/gemini-2.5-pro-exp-03-25", "microsoft/phi-4-multimodal-instruct",
        "anthropic/claude-3.7-sonnet:beta", "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet:beta", "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku:beta", "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-haiku:beta"
    ]

    logger.info(f"Processing query: '{shorten(question, 100)}'")
    logger.info(f"Using model: {preferred_model}")

    image_attachments = []
    if image_url:
        logger.info("Processing image URL...")
        base64_image = await _download_image_to_base64(image_url)
        if base64_image:
            image_attachments.append(base64_image)
        else:
            logger.error("Failed to process image URL. Proceeding without image.")

    # --- Search ---
    logger.info("Searching documents...")
    try:
        # Use simple search, no conversation context in CLI
        # Apply reranking based on config
        search_results = await doc_manager.search(
            question,
            top_k=config.get_top_k_for_model(preferred_model),
            apply_reranking=config.RERANKING_ENABLED
        )
        logger.info(f"Found {len(search_results)} relevant document sections.")
    except Exception as e:
        logger.error(f"Error during document search: {e}", exc_info=True)
        search_results = []

    # --- Format Context ---
    raw_doc_contexts = []
    image_ids_from_search = [] # Track images found in search results
    googledoc_mapping = doc_manager.get_original_name_to_googledoc_id_mapping() # Get mapping for citations

    for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
        if image_id:
            # This is an image description
            image_name = image_manager.metadata.get(image_id, {}).get('name', "Unknown Image")
            raw_doc_contexts.append(f"Image: {image_name} (ID: {image_id})\nDescription: {chunk}\nRelevance: {score:.2f}")
            if image_id not in image_ids_from_search:
                image_ids_from_search.append(image_id)
        elif doc in googledoc_mapping:
            # Create citation link for Google Doc
            doc_id = googledoc_mapping[doc]
            doc_url = f"https://docs.google.com/document/d/{doc_id}/"
            raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) [Citation URL: {doc_url}] (similarity: {score:.2f}):\n{chunk}")
        else:
            raw_doc_contexts.append(f"From document '{doc}' (Chunk {chunk_index}/{total_chunks}) (similarity: {score:.2f}):\n{chunk}")

    # --- Prepare Messages for LLM ---
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] # Basic system prompt

    # Add raw document context
    if raw_doc_contexts:
        raw_doc_context_combined = "\n\n".join(raw_doc_contexts)
        # Truncate if needed
        max_raw_context_len = 30000
        if len(raw_doc_context_combined) > max_raw_context_len:
            raw_doc_context_combined = raw_doc_context_combined[:max_raw_context_len] + "\n... [Context Truncated]"
            logger.warning(f"Raw document context truncated to {max_raw_context_len} characters.")
        messages.append({
            "role": "system",
            "content": f"Raw document context (with citation links):\n{raw_doc_context_combined}"
        })

    # Add image context summary (if applicable)
    total_images = len(image_attachments) + len(image_ids_from_search)
    if total_images > 0:
        img_source_parts = []
        if image_attachments: img_source_parts.append(f"{len(image_attachments)} attached")
        if image_ids_from_search: img_source_parts.append(f"{len(image_ids_from_search)} from search")
        messages.append({
            "role": "system",
            "content": f"The query context includes {total_images} image{'s' if total_images > 1 else ''} ({', '.join(img_source_parts)}). Vision models will see these in the user message."
        })

    # Add the user's question
    messages.append({"role": "user", "content": question})

    # --- Call LLM ---
    logger.info("Sending request to LLM...")
    # Use a default temperature for CLI, maybe slightly lower?
    temperature = config.TEMPERATURE_BASE * 0.9

    # Combine image attachments and search images for the API call
    all_api_images = image_attachments # Start with attached images
    if preferred_model in vision_capable_models and image_ids_from_search: # Check preferred_model here for logging intent
        logger.info("Adding images found during search to API call (if a vision model is used)...")
        # The actual adding happens inside _cli_try_ai_completion if the current model is vision capable
        for img_id in image_ids_from_search:
            try:
                base64_image = image_manager.get_base64_image(img_id)
                if base64_image:
                    all_api_images.append(base64_image) # Add to the list passed to the completion function
                else:
                     logger.warning(f"Could not retrieve base64 for search image ID: {img_id}")
            except Exception as e:
                logger.error(f"Error getting base64 for search image ID {img_id}: {e}")

    # Call the updated completion function
    completion_result, actual_model_used = await _cli_try_ai_completion( # Capture both return values
        config=config,
        initial_model=preferred_model, # Pass the initially preferred model
        messages=messages,
        image_attachments=all_api_images, # Pass combined list
        temperature=temperature,
        timeout_duration=config.API_TIMEOUT # Use API_TIMEOUT from config
    )

    # --- Output Response ---
    if actual_model_used: # Check if a model succeeded
        logger.info(f"Response generated using model: {actual_model_used}")
        # The validation is now done inside _cli_try_ai_completion
        response = completion_result['choices'][0]['message']['content']
        # Print the response
        print("\n--- Publicia Response ---")
        print(response)
        print("-----------------------")
    else:
        # This branch is hit if _cli_try_ai_completion returned (None, None)
        logger.error("Failed to get response from LLM after trying all fallbacks.")
        print("\nError: Failed to get response from AI after trying available models.", file=sys.stderr)


async def run_cli():
    """Parses arguments, initializes, and runs the query."""
    parser = argparse.ArgumentParser(description="Query the Publicia bot from the command line.")
    parser.add_argument("question", help="The question to ask Publicia.")
    parser.add_argument("--image-url", help="Optional URL to an image to analyze.", default=None)
    parser.add_argument("--model", help="Optional specific model to use (e.g., 'google/gemini-2.5-flash-preview').", default=None)
    # Add a debug flag?
    # parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    # Initialize managers
    config, doc_manager, image_manager, pref_manager = await initialize_managers()

    # Process the query
    await process_cli_query(args, config, doc_manager, image_manager, pref_manager)

async def run_convert_gdocs_format():
    """Initializes DocumentManager and runs the GDoc conversion utility."""
    logger.info("Starting Google Docs tracking file conversion utility...")
    config, doc_manager, _, _ = await initialize_managers()
    
    # Ensure documents (and thus metadata) are loaded before attempting conversion
    if not doc_manager.metadata:
        logger.info("Metadata not loaded, attempting to load documents first...")
        await doc_manager._load_documents() # This loads metadata
        if not doc_manager.metadata:
            logger.error("Failed to load document metadata. Cannot proceed with conversion.")
            print("Error: Failed to load document metadata. Conversion aborted.", file=sys.stderr)
            return

    logger.info("Calling convert_tracked_gdocs_to_uuid_format...")
    success = await doc_manager.convert_tracked_gdocs_to_uuid_format()
    if success:
        print("Google Docs tracking file conversion process completed successfully.")
        logger.info("Google Docs tracking file conversion process completed successfully.")
    else:
        print("Google Docs tracking file conversion process failed. Check logs for details.", file=sys.stderr)
        logger.error("Google Docs tracking file conversion process failed.")

if __name__ == "__main__":
    # Check for a special argument to run the conversion
    if "--convert-gdocs-format" in sys.argv:
        sys.argv.remove("--convert-gdocs-format") # Remove it so argparse doesn't see it
        try:
            asyncio.run(run_convert_gdocs_format())
        except KeyboardInterrupt:
            logger.info("GDoc conversion interrupted by user.")
            print("\nConversion operation cancelled.", file=sys.stderr)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during GDoc conversion: {e}", exc_info=True)
            print(f"\nFatal Error during GDoc conversion: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            asyncio.run(run_cli())
        except KeyboardInterrupt:
            logger.info("CLI execution interrupted by user.")
            print("\nOperation cancelled.", file=sys.stderr)
        except Exception as e:
            logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
            print(f"\nFatal Error: {e}", file=sys.stderr)
            sys.exit(1)

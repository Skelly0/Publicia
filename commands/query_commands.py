"""
Query commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import re
import aiohttp
import asyncio
import base64
import json
import os
import urllib.parse # Moved import to top level
from datetime import datetime, timedelta, timezone # Added timedelta, timezone
from utils.helpers import split_message, check_permissions, xml_wrap, wrap_document
from utils.logging import log_qa_pair
# Import both system prompts and the new functions
from prompts.system_prompt import SYSTEM_PROMPT, INFORMATIONAL_SYSTEM_PROMPT, get_system_prompt_with_documents, get_informational_system_prompt_with_documents

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all query commands with the bot."""

    @bot.tree.command(name="query", description="Ask Publicia a question about Ledus Banum 77 and Imperial lore")
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
                logger.error(f"Error deferring interaction: {e}") # Correctly indented under inner except
                return  # Exit on any other error

            # Log user info after successful deferral
            logger.info(f"Query received from user: {interaction.user.name} (ID: {interaction.user.id})") # Correctly indented under outer try

            if not question: # Correctly indented under outer try
                await interaction.followup.send("*neural error detected!* Please provide a question.") # Correctly indented under if
                return # Correctly indented under if

            # Get channel name and user info
            channel_name = interaction.channel.name if interaction.guild else "DM" # Correctly indented under outer try
            nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name # Correctly indented under outer try
            
            # Process image URL if provided
            image_attachments = [] # This will hold tuples of (bytes, base64_string)
            status_message = None
            
            if image_url:
                try:
                    if any(image_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        status_message = await interaction.followup.send("*neural pathways activating... analyzing query and image...*", ephemeral=False)
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                                if resp.status == 200:
                                    content_type = resp.headers.get('Content-Type', 'image/jpeg')
                                    if content_type.startswith('image/'):
                                        image_data = await resp.read()
                                        base64_encoded = base64.b64encode(image_data).decode('utf-8')
                                        base64_string = f"data:{content_type};base64,{base64_encoded}"
                                        image_attachments.append((image_data, base64_string)) # Append tuple
                                        logger.debug(f"Processed image from URL: {image_url}")
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
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id), 
                default_model=bot.config.LLM_MODEL
            )

            # Use the hybrid search system
            await status_message.edit(content="*analyzing query and searching imperial databases...*")
            search_results = await bot.process_hybrid_query(  # Await async call
                question,
                nickname,  # Use nickname instead of interaction.user.name for better search context
                max_results=bot.config.get_top_k_for_model(preferred_model),
                use_context=False,  # this disables all the context logic for slash commands
            )

            # Determine how many unique documents were returned
            doc_count = len({doc_uuid for doc_uuid, *_ in search_results})

            chunk_details = [
                f"{name}:{idx}"
                for _, name, _, _, _, idx, _ in search_results
            ]

            # Log the results
            logger.info(
                f"Found {len(search_results)} relevant document sections from {doc_count} documents"
            )
            
            await status_message.edit(content="*synthesizing information...*")
            
            # Load Google Doc ID mapping for citation links
            googledoc_mapping = bot.document_manager.get_original_name_to_googledoc_id_mapping()

            # Extract image IDs from search results
            image_ids = []
            # Assuming search_results returns (doc_uuid, original_name, chunk, score, image_id, chunk_index, total_chunks)
            for doc_uuid, original_name, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
                    logger.debug(f"Found relevant image: {image_id}")
            
            # Check if the question contains any Google Doc links
            doc_ids = await bot._extract_google_doc_ids(question)
            google_doc_contents = []
            if doc_ids:
                await status_message.edit(content="*detected Google Doc links in your query... fetching content...*")
                for doc_id, doc_url in doc_ids:
                    content = await bot._fetch_google_doc_content(doc_id)
                    if content:
                        google_doc_contents.append((doc_id, doc_url, content))

            # Format raw results with citation info
            # import urllib.parse # Removed from here
            raw_doc_contexts = []
            # Assuming search_results returns (doc_uuid, original_name, chunk, score, image_id,
            #                                  chunk_index, total_chunks)
            for doc_uuid, original_name, chunk, score, image_id, chunk_index, total_chunks in search_results:

                # Pre-amble for “Region” docs
                preamble = "NOTE: The following chunk is about **natives** and **not the Empire**.\n" \
                        if "region" in original_name.lower() else ""

                if image_id:
                    image_name = (
                        bot.image_manager.metadata[image_id]["name"]
                        if image_id in bot.image_manager.metadata
                        else "Unknown Image"
                    )
                    raw_doc_contexts.append(
                        wrap_document(
                            f"{preamble}{chunk}",
                            f"Image: {image_name} (ID: {image_id})",
                            metadata=f"similarity: {score:.2f}"
                        )
                    )

                elif original_name in googledoc_mapping:
                    doc_id = googledoc_mapping[original_name]
                    words = chunk.split()
                    search_text = " ".join(words[: min(10, len(words))])
                    encoded_search = urllib.parse.quote(search_text)
                    doc_url = f"https://docs.google.com/document/d/{doc_id}/"
                    raw_doc_contexts.append(
                        wrap_document(
                            f"{preamble}{chunk}",
                            f"{original_name} (Chunk {chunk_index}/{total_chunks})",
                            metadata=f"url: {doc_url}; similarity: {score:.2f}"
                        )
                    )

                else:
                    raw_doc_contexts.append(
                        wrap_document(
                            f"{preamble}{chunk}",
                            f"{original_name} (Chunk {chunk_index}/{total_chunks})",
                            metadata=f"similarity: {score:.2f}"
                        )
                    )

            # Add fetched Google Doc content to context
            google_doc_context = []
            for doc_id, doc_url, content in google_doc_contents:
                truncated_content = content[:2000] + ("..." if len(content) > 2000 else "")
                google_doc_context.append(
                    wrap_document(truncated_content, doc_url)
                )

            # Fetch user pronouns
            pronouns = bot.user_preferences_manager.get_pronouns(str(interaction.user.id))

            user_info_message = {
                "role": "system",
                "content": xml_wrap(
                    "user_information",
                    f"User Information: The users nickname is: {nickname}."
                ),
            }

            pronoun_context_message = None
            if pronouns:
                logger.debug(
                    f"User {interaction.user.id} ({nickname}) has pronouns set: {pronouns}"
                )
                pronoun_context_message = {
                    "role": "system",
                    "content": xml_wrap(
                        "user_pronouns",
                        f"""The user provided this pronoun string: \"{pronouns}\".\n\n"
                        "Your job:\n"
                        "1. split that string on “/” into segments.\n"
                        "    - subject = segment[0]\n"
                        "    - object  = segment[1] if it exists, else subject\n"
                        "    - possessive = segment[2] if it exists, else object\n"
                        "2. whenever you talk *about* the player in third-person, use those pronouns.\n"
                        "3. when you talk directly *to* the player, always say “you.”\n"
                        "4. do NOT echo the literal pronouns string, or the parsing instructions, in your dialogue.\n"
                        "5. do NOT reference the pronouns directly, work them in naturally\n"
                        "if parsing fails, fall back to they/them/theirs."
                        """
                    ),
                }
            else:
                logger.debug(
                    f"User {interaction.user.id} ({nickname}) has no pronouns set."
                )

            # Get document list content
            document_list_content = bot.document_manager.get_document_list_content()
            
            # Determine which system prompt to use based on user preference
            use_informational_prompt = bot.user_preferences_manager.get_informational_prompt_mode(str(interaction.user.id))
            if use_informational_prompt:
                selected_system_prompt = get_informational_system_prompt_with_documents(document_list_content)
            else:
                selected_system_prompt = get_system_prompt_with_documents(document_list_content)
            logger.debug(f"Using {'Informational' if use_informational_prompt else 'Standard'} System Prompt with document list for user {interaction.user.id} in /query command")

            messages = []

            raw_doc_context = "\n\n".join(raw_doc_contexts)
            if raw_doc_context:
                messages.append(
                    {
                        "role": "system",
                        "content": xml_wrap(
                            "document_context",
                            f"Raw document context (with citation links):\n{raw_doc_context}",
                        ),
                    }
                )

            if google_doc_context:
                google_docs_content = "\n\n".join(google_doc_context)
                messages.append(
                    {
                        "role": "system",
                        "content": xml_wrap(
                            "google_docs_context",
                            f"Content from Google Docs linked in the query:\n\n{google_docs_content}",
                        ),
                    }
                )

            messages.append(
                {
                    "role": "system",
                    "content": selected_system_prompt,
                }
            )

            messages.append(user_info_message)

            if pronoun_context_message:
                messages.append(pronoun_context_message)

            # --- Add Keyword Context ---
            found_keywords_in_chunks = set()
            if search_results:
                # Limit the number of chunks to check based on config
                limit = bot.config.KEYWORD_CHECK_CHUNK_LIMIT
                logger.debug(f"Scanning up to {limit} search result chunks for keywords in /query...")
                # Unpack 7 items, ignoring those not used in this loop
                for i, (_, _, chunk, _, _, _, _) in enumerate(search_results):
                    if i >= limit:
                        logger.debug(f"Reached keyword check limit ({limit}), stopping scan.")
                        break # Stop checking after reaching the limit
                    # Use bot.keyword_manager here
                    keywords_in_chunk = bot.keyword_manager.find_keywords_in_text(chunk)
                    if keywords_in_chunk:
                        found_keywords_in_chunks.update(keywords_in_chunk)
                if found_keywords_in_chunks:
                    logger.debug(f"Found keywords in search chunks: {', '.join(found_keywords_in_chunks)}")
                else:
                    logger.debug("No keywords found in search chunks.")

            if found_keywords_in_chunks:
                keyword_context_parts = []
                definitions_count = 0 # Track total definitions added
                for keyword in found_keywords_in_chunks:
                    # get_info_for_keyword now returns Optional[List[str]]
                    definitions = bot.keyword_manager.get_info_for_keyword(keyword)
                    if definitions: # Check if the list is not None and not empty
                        for definition in definitions:
                            keyword_context_parts.append(f"- {keyword.capitalize()}: {definition}")
                            definitions_count += 1

                if keyword_context_parts:
                    # Adjust the introductory text slightly if needed
                    keyword_context_str = f"Additional Context from Keyword Database ({definitions_count} entries found, duplicates possible):\n" + "\n".join(keyword_context_parts)
                    # Truncate if necessary
                    max_keyword_context_len = 8192 # Adjust as needed
                    if len(keyword_context_str) > max_keyword_context_len:
                         keyword_context_str = keyword_context_str[:max_keyword_context_len] + "\n... [Keyword Context Truncated]"
                         logger.warning(f"Keyword context truncated to {max_keyword_context_len} characters.")

                    messages.append({
                        "role": "system",
                        "content": xml_wrap("keyword_context", keyword_context_str),
                    })
                    logger.debug(f"Added context for {definitions_count} keyword definitions (from {len(found_keywords_in_chunks)} unique keywords) to /query.")
            # --- End Keyword Context ---

            # Add the query itself
            messages.append({
                "role": "user",
                "content": xml_wrap(
                    "channel_info",
                    f"You are responding to a message in the Discord channel: {channel_name}",
                ),
            })
            
            # --- Vision Fallback Handling ---
            # This function will generate descriptions if the model is not vision-capable
            # and modify the messages list accordingly. It returns the final images to be sent to the API.
            if image_attachments and preferred_model not in bot.vision_capable_models:
                await status_message.edit(content=f"*Your preferred model model doesn't support images. Using a vision model to generate descriptions first...*")

            messages, api_image_attachments = await bot._handle_image_vision_fallback(
                preferred_model,
                image_attachments, # This is the list of (bytes, base64) tuples
                messages
            )
            # --- End Vision Fallback Handling ---

            # Add image context if there are images from search or attachments
            if image_ids or api_image_attachments:
                total_images = len(image_ids) + len(api_image_attachments)
                img_source = []
                if image_ids:
                    img_source.append(f"{len(image_ids)} from search results")
                if api_image_attachments:
                    img_source.append(f"{len(api_image_attachments)} from attachments")
                
                messages.append({
                    "role": "system",
                    "content": xml_wrap(
                        "image_summary",
                        f"The query has {total_images} relevant images ({', '.join(img_source)}). If you are a vision-capable model, you will see these images in the user's message.",
                    ),
                })

            messages.append({
                "role": "user",
                "content": f"{question}"
            })

            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            if "deepseek/deepseek-r1" in preferred_model:
                model_name = "DeepSeek-R1"
            elif "meta-llama/llama-4-maverick" in preferred_model:
                model_name = "Llama 4 Maverick"
            elif "deepseek/deepseek-chat-v3" in preferred_model:
                model_name = "DeepSeek V3 0324"
            elif "gemini-2.5-flash" in preferred_model:
                model_name = "Gemini 2.5 Flash"
            elif "google/gemini-2.5-pro" in preferred_model:
                model_name = "Gemini 2.5 Pro"
            elif preferred_model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in preferred_model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-sonnet-4" in preferred_model:
                model_name = "Claude 4 Sonnet"
            elif "claude-3.7-sonnet" in preferred_model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in preferred_model:
                model_name = "Qwen QwQ 32B"
            elif "qwen/qwen3-235b-a22b-07-25" in preferred_model:
                model_name = "Qwen 3 235B A22B"
            elif "moonshotai/kimi-k2" in preferred_model:
                model_name = "Kimi K2"
            elif "switchpoint/router" in preferred_model:
                model_name = "Switchpoint Router"
            elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in preferred_model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in preferred_model:
                model_name = "Anubis Pro 105B"
            elif "microsoft/phi-4-multimodal-instruct" in preferred_model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in preferred_model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in preferred_model:
                model_name = "Phi-3.5 Mini"
            elif "grok" in preferred_model:
                model_name = "Grok 3 Mini"
            elif preferred_model == "openai/gpt-4.1-mini":
                model_name = "OpenAI GPT-4.1 Mini"
            elif preferred_model == "openai/gpt-4.1-nano":
                model_name = "OpenAI GPT-4.1 Nano"
            elif preferred_model == "openai/o4-mini":
                model_name = "OpenAI o4 Mini"
            elif preferred_model == "minimax/minimax-m1":
                model_name = "MiniMax M1"

            if (image_attachments or image_ids) and preferred_model not in bot.vision_capable_models:
                await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*\nusing image descriptions generated by a vision-capable model as the currently select model is not vision-capable.")
            else:
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(fetched content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})*")
                else:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")

            # Check if channel parsing is enabled for this user
            channel_parsing_enabled, channel_parsing_count = bot.user_preferences_manager.get_channel_parsing_settings(
                str(interaction.user.id)
            )

            # If enabled, fetch channel messages and add to context
            if channel_parsing_enabled and channel_parsing_count > 0:
                await status_message.edit(content="*analyzing channel conversation context...*")
                
                # Fetch channel messages
                channel_messages = await bot.fetch_channel_messages(
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
                        "content": xml_wrap("channel_context", channel_context),
                    })
                    
                    logger.debug(f"Added {len(channel_messages)} channel messages to context")
                    await status_message.edit(content=f"*analyzing query, search results, and channel context ({len(channel_messages)} messages)...*")
                    
            temperature, t_min, t_base, t_max = bot.calculate_dynamic_temperature(
                question,
                user_id=str(interaction.user.id)
            )

            completion, actual_model = await bot._try_ai_completion(
                preferred_model,
                messages,
                image_ids=image_ids,
                image_attachments=api_image_attachments, # Use the (potentially empty) list from the fallback handler
                temperature=temperature 
            )

            if completion and completion.get('choices') and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                    response = completion['choices'][0]['message']['content']
                    
                    # Pass the status message as the existing_message parameter
                    await bot.send_split_message(
                        interaction.channel,
                        response,
                        model_used=actual_model,
                        user_id=str(interaction.user.id),
                        existing_message=status_message
                    )
                    total_chunks = sum(
                        len(chunks) for chunks in bot.document_manager.chunks.values()
                    )
                    context_info = {
                        "reply": False,
                        "direct_images": 0,
                        "reply_images": 0,
                        "search_images": 0,
                        "google_docs": 0,
                        "chunks": total_chunks,
                        "chunk_details": chunk_details,
                        "channel_messages": 0,
                        "doc_count": doc_count,
                        "temperature_min": t_min,
                        "temperature_base": t_base,
                        "temperature_max": t_max,
                        "temperature_used": temperature,
                    }
                    log_qa_pair(
                        question,
                        response,
                        interaction.user.name,
                        channel_name,
                        multi_turn=False,
                        interaction_type="slash_command",
                        context=context_info,
                        model_used=actual_model,
                        temperature=temperature,
                        temperature_min=t_min,
                        temperature_base=t_base,
                        temperature_max=t_max,
                    )
                else:
                    logger.error(f"Unexpected response structure: {completion}")
                    await status_message.edit(content="*neural circuit overload!* I received an unexpected response structure.")
            else:
                await interaction.followup.send("*synaptic failure detected!* I apologize, but I'm having trouble generating a response right now.")

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True) # Added exc_info for better debugging
            try:
                # Check if status_message exists before trying to edit
                if 'status_message' in locals() and status_message and isinstance(status_message, discord.WebhookMessage): # Check type too
                    await status_message.edit(content="*neural circuit overload!* My brain is struggling and an error has occurred.")
                else:
                    # If defer failed or status_message wasn't created, send a new message via followup
                    await interaction.followup.send("*neural circuit overload!* My brain is struggling and an error has occurred.")
            except Exception as send_error:
                logger.error(f"Failed to send error message to user: {send_error}")


    @bot.tree.command(name="query_full_context", description="Ask Publicia a question using ALL documents as context (1/day limit)")
    @app_commands.describe(
        question="Your question using the full document context"
    )
    async def query_full_context(interaction: discord.Interaction, question: str):
        user_id_str = str(interaction.user.id)
        user_name = interaction.user.name
        logger.info(f"Full context query received from user: {user_name} (ID: {user_id_str})")

        # --- Check if user is admin (bypass limit if so) using helper ---
        is_admin = await check_permissions(interaction)
        if is_admin:
            logger.debug(f"User {user_name} (ID: {user_id_str}) has admin privileges. Bypassing usage limit.")
        else:
             logger.debug(f"User {user_name} (ID: {user_id_str}) does not have admin privileges. Applying usage limit.")


        # --- Daily Usage Limit Check (Skip for Admins) ---
        can_use = True
        time_remaining_str = ""
        last_usage_str = None # Initialize to ensure it exists
        if not is_admin:
            last_usage_str = bot.user_preferences_manager.get_last_full_context_usage(user_id_str)
        if last_usage_str: # Now this check is safe
            try:
                last_usage_time = datetime.fromisoformat(last_usage_str).replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                if now_utc - last_usage_time < timedelta(days=1):
                    can_use = False
                    next_available_time = last_usage_time + timedelta(days=1)
                    time_remaining = next_available_time - now_utc
                    # Format time remaining
                    hours, remainder = divmod(time_remaining.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    time_remaining_str = f"{int(hours)}h {int(minutes)}m"

            except ValueError:
                logger.error(f"Could not parse last usage timestamp '{last_usage_str}' for user {user_id_str}. Allowing usage.")
            except Exception as e:
                 logger.error(f"Error checking usage limit for user {user_id_str}: {e}. Allowing usage.")
        # End of non-admin check block

        if not is_admin and not can_use: # Apply limit only if not admin and cannot use
            await interaction.response.send_message(
                f"*Usage limit reached!* The full context query can only be used once per day (admins excluded). "
                f"Please try again in approximately {time_remaining_str}.",
                ephemeral=True
            )
            logger.warning(f"User {user_id_str} attempted to use query_full_context but hit the daily limit.")
            return

        # --- Proceed with Command ---
        try:
            await interaction.response.defer()
        except discord.errors.NotFound:
            logger.warning("Interaction expired before deferral for query_full_context")
            return
        except Exception as e:
            logger.error(f"Error deferring interaction for query_full_context: {e}")
            return

        status_message = await interaction.followup.send("*neural pathways activating... gathering all documents...*", ephemeral=False)

        try:
            # Get user info
            channel_name = interaction.channel.name if interaction.guild else "DM"
            nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name

            # --- Get All Document Content ---
            await status_message.edit(content="*retrieving full content of all managed documents... this may take a moment...*")
            all_doc_contents = bot.document_manager.get_all_document_contents() # Synchronous call

            if not all_doc_contents:
                await status_message.edit(content="*neural error detected!* No documents found to provide context.")
                return

            # --- Combine and Estimate Token Count ---
            await status_message.edit(content=f"*processing content from {len(all_doc_contents)} documents... estimating size...*")
            full_context_str = ""
            total_chars = 0
            for name, content in all_doc_contents.items():
                full_context_str += wrap_document(content, name) + "\n\n"
                total_chars += len(content)

            chunk_details = [f"{name}:ALL" for name in all_doc_contents.keys()]

            # Estimate tokens (simple heuristic: 4 chars/token)
            estimated_tokens = total_chars // 4
            # Define model context limit (adjust as needed for Gemini 2.5 Pro)
            # Let's target slightly under 1M tokens for safety margin with Gemini 2.5 Pro Experimental
            TOKEN_LIMIT = 800000
            truncated = False

            logger.debug(f"Estimated total characters: {total_chars}, Estimated tokens: {estimated_tokens}")

            # --- Truncate if Necessary ---
            if estimated_tokens > TOKEN_LIMIT:
                truncated = True
                await status_message.edit(content=f"*warning: combined document content ({estimated_tokens} tokens) exceeds limit ({TOKEN_LIMIT}). Truncating...*")
                logger.warning(f"Full context ({estimated_tokens} tokens) exceeds limit ({TOKEN_LIMIT}). Truncating.")

                # Simple truncation: keep only enough characters from the start
                max_chars = TOKEN_LIMIT * 4 # Target character count
                full_context_str = full_context_str[:max_chars] + "\n\n... (Content truncated due to length)"
                estimated_tokens = TOKEN_LIMIT # Update estimated tokens after truncation

            # --- Prepare Messages for Model ---
            await status_message.edit(content="*preparing context for the advanced model...*")

            # Fetch user pronouns
            pronouns = bot.user_preferences_manager.get_pronouns(user_id_str)

            user_info_message = {
                "role": "system",
                "content": xml_wrap(
                    "user_information",
                    f"User Information: The users character name/nickname is: {nickname}."
                ),
            }

            pronoun_context_message = None
            if pronouns:
                pronoun_context_message = {
                    "role": "system",
                    "content": xml_wrap(
                        "user_pronouns",
                        f"""The user provided this pronoun string: \"{pronouns}\".\n\n"
                        "Your job:\n"
                        "1. split that string on “/” into segments.\n"
                        "    - subject = segment[0]\n"
                        "    - object  = segment[1] if it exists, else subject\n"
                        "    - possessive = segment[2] if it exists, else object\n"
                        "2. whenever you talk *about* the player in third-person, use those pronouns.\n"
                        "3. when you talk directly *to* the player, always say “you.”\n"
                        "4. do NOT echo the literal pronouns string, or the parsing instructions, in your dialogue.\n"
                        "5. do NOT reference the pronouns directly, work them in naturally\n"
                        "if parsing fails, fall back to they/them/theirs."
                        """
                    ),
                }
            else:
                logger.debug(f"User {nickname} has no pronouns set.")

            # Get document list content and use standard system prompt with documents
            document_list_content = bot.document_manager.get_document_list_content()
            selected_system_prompt = get_system_prompt_with_documents(document_list_content)

            messages = []

            messages.append(
                {
                    "role": "system",
                    "content": xml_wrap(
                        "document_context_full",
                        f"Full document context follows {'(truncated)' if truncated else ''}:\n\n{full_context_str}",
                    ),
                }
            )

            messages.append({"role": "system", "content": selected_system_prompt})

            messages.append(user_info_message)

            if pronoun_context_message:
                messages.append(pronoun_context_message)

            messages.append(
                {
                    "role": "user",
                    "content": xml_wrap(
                        "channel_info",
                        f"You are responding to a message in the Discord channel: {channel_name}",
                    ),
                }
            )

            messages.append({"role": "user", "content": f"{nickname}: {question}"})

            # --- Model Selection and Execution ---
            # Specify the exact models to use
            target_models = ["google/gemini-2.5-flash:thinking", "google/gemini-2.5-flash", "cohere/command-r7b-12-2024"]#, "google/gemini-2.5-pro"]
            model_name_display = "google/gemini-2.5-flash:thinking" # Friendly name

            await status_message.edit(content=f"*formulating response using {model_name_display}...*")

            temperature, t_min, t_base, t_max = bot.calculate_dynamic_temperature(
                question,
                user_id=str(interaction.user.id)
            )  # Use dynamic temp

            # Log the models being attempted
            logger.debug(f"Attempting full context query with models: {target_models}")

            # Call _try_ai_completion - Assuming it can handle a list of models to try
            # If not, bot.py needs adjustment. For now, passing the list.
            completion, actual_model = await bot._try_ai_completion(
                target_models, # Pass the list of models
                messages,
                temperature=temperature
                # No image handling needed for this command
            )

            # Log the model that was actually used
            logger.info(f"Full context query completed using model: {actual_model}")

            # --- Handle Response ---
            if completion and completion.get('choices') and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                    response = completion['choices'][0]['message']['content']

                    # Record usage *after* successful generation
                    bot.user_preferences_manager.record_full_context_usage(user_id_str)
                    logger.debug(f"Recorded full context usage for user {user_id_str}")

                    await bot.send_split_message(
                        interaction.channel,
                        response,
                        model_used=actual_model or target_models[0], # Show model used
                        user_id=user_id_str,
                        existing_message=status_message
                    )
                    total_chunks = sum(len(chunks) for chunks in bot.document_manager.chunks.values())
                    context_info = {
                        "reply": False,
                        "direct_images": 0,
                        "reply_images": 0,
                        "search_images": 0,
                        "google_docs": 0,
                        "chunks": total_chunks,
                        "chunk_details": chunk_details,
                        "channel_messages": 0,
                        "doc_count": len(all_doc_contents),
                        "temperature_min": t_min,
                        "temperature_base": t_base,
                        "temperature_max": t_max,
                        "temperature_used": temperature,
                    }
                    log_qa_pair(
                        question,
                        response,
                        interaction.user.name,
                        channel_name,
                        multi_turn=False,
                        interaction_type="slash_command",
                        context=context_info,
                        model_used=actual_model or target_models[0],
                        temperature=temperature,
                        temperature_min=t_min,
                        temperature_base=t_base,
                        temperature_max=t_max,
                    )
                else:
                    logger.error(f"Unexpected response structure from full context query: {completion}")
                    await status_message.edit(content="*neural circuit overload!* Received an unexpected response structure.")
            else:
                # Don't record usage if generation failed
                await status_message.edit(content="*synaptic failure detected!* Apologies, the advanced model failed to generate a response.")

        except Exception as e:
            logger.error(f"Error processing full context query: {e}", exc_info=True)
            try:
                # Check if status_message exists before trying to edit
                if 'status_message' in locals() and status_message:
                    await status_message.edit(content="*neural circuit overload!* My brain is struggling and an error has occurred during the full context query.")
                else:
                    # If defer failed, status_message might not exist
                    await interaction.followup.send("*neural circuit overload!* My brain is struggling and an error has occurred during the full context query.")
            except Exception as send_error:
                logger.error(f"Failed to send error message to user during full context query: {send_error}")

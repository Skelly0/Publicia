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
from datetime import datetime
from utils.helpers import split_message
from prompts.system_prompt import SYSTEM_PROMPT

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
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id), 
                default_model=bot.config.LLM_MODEL
            )

            # Use the hybrid search system
            await status_message.edit(content="*analyzing query and searching imperial databases...*")
            search_results = await bot.process_hybrid_query( # Await async call
                question,
                interaction.user.name,
                max_results=bot.config.get_top_k_for_model(preferred_model),
                use_context=False  # this disables all the context logic for slash commands
            )
            
            # Log the results
            logger.info(f"Found {len(search_results)} relevant document sections")
            
            await status_message.edit(content="*synthesizing information...*")
            
            # Load Google Doc ID mapping for citation links
            googledoc_mapping = bot.document_manager.get_googledoc_id_mapping()

            # Extract image IDs from search results
            image_ids = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
                    logger.info(f"Found relevant image: {image_id}")
            
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
            import urllib.parse
            raw_doc_contexts = []
            for doc, chunk, score, image_id, chunk_index, total_chunks in search_results:
                if image_id:
                    # This is an image description
                    image_name = bot.image_manager.metadata[image_id]['name'] if image_id in bot.image_manager.metadata else "Unknown Image"
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
            elif "deepseek/deepseek-chat-v3" in preferred_model:
                model_name = "DeepSeek V3 0324"
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
            elif "microsoft/phi-4-multimodal-instruct" in preferred_model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in preferred_model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in preferred_model:
                model_name = "Phi-3.5 Mini"

            if (image_attachments or image_ids) and preferred_model not in bot.vision_capable_models:
                await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*\n*note: your preferred model ({model_name}) doesn't support image analysis. only the text content will be processed.*")
            else:
                # Add a note about fetched Google Docs if any were processed
                if google_doc_contents:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...\n(fetched content from {len(google_doc_contents)} linked Google Doc{'s' if len(google_doc_contents) > 1 else ''})*")
                else:
                    await status_message.edit(content=f"*formulating response with enhanced neural mechanisms using {model_name}...*")

            # Check if channel parsing is enabled for this user
            channel_parsing_enabled, channel_parsing_count = bot.user_preferences_manager.get_channel_parsing(
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
                        "content": channel_context
                    })
                    
                    logger.info(f"Added {len(channel_messages)} channel messages to context")
                    await status_message.edit(content=f"*analyzing query, search results, and channel context ({len(channel_messages)} messages)...*")
                    
            temperature = bot.calculate_dynamic_temperature(
                question
            )

            completion, actual_model = await bot._try_ai_completion(
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
                    await bot.send_split_message(
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

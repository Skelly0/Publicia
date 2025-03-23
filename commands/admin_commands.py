"""
Admin-only commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import os
import aiohttp
import base64
from datetime import datetime
from utils.helpers import check_permissions
from prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all admin commands with the bot."""
    
    @bot.tree.command(name="compare_models", description="Compare responses from multiple AI models (admin only)")
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
                    is_vision_model = model in bot.vision_capable_models
                    use_images = is_vision_model and image_attachments
                    
                    # Use the try_ai_completion method
                    completion, actual_model = await bot._try_ai_completion(
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
                        "is_vision_model": model in bot.vision_capable_models,
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
    Timeout per model: {bot.config.API_TIMEOUT} seconds (from config)
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

    @bot.tree.command(name="ban_user", description="Ban a user from using the bot (admin only)")
    @app_commands.describe(user="User to ban")
    @app_commands.check(check_permissions)
    async def ban_user(interaction: discord.Interaction, user: discord.User):
        await interaction.response.defer()
        if user.id in bot.banned_users:
            await interaction.followup.send(f"{user.name} is already banned.")
        else:
            bot.banned_users.add(user.id)
            bot.save_banned_users()
            await interaction.followup.send(f"Banned {user.name} from using the bot.")
            logger.info(f"User {user.name} (ID: {user.id}) banned by {interaction.user.name}")

    @bot.tree.command(name="unban_user", description="Unban a user (admin only)")
    @app_commands.describe(user="User to unban")
    @app_commands.check(check_permissions)
    async def unban_user(interaction: discord.Interaction, user: discord.User):
        await interaction.response.defer()
        if user.id not in bot.banned_users:
            await interaction.followup.send(f"{user.name} is not banned.")
        else:
            bot.banned_users.remove(user.id)
            bot.save_banned_users()
            await interaction.followup.send(f"Unbanned {user.name}.")
            logger.info(f"User {user.name} (ID: {user.id}) unbanned by {interaction.user.name}")

    @bot.tree.command(name="reload_docs", description="Reload all documents from disk (admin only)")
    @app_commands.check(check_permissions)
    async def reload_docs(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            await bot.document_manager.reload_documents()
            await interaction.followup.send("Documents reloaded successfully.")
        except Exception as e:
            await interaction.followup.send(f"Error reloading documents: {str(e)}")

    @bot.tree.command(name="regenerate_embeddings", description="Regenerate all document embeddings (admin only)")
    @app_commands.check(check_permissions)
    async def regenerate_embeddings(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            status_message = await interaction.followup.send("*starting neural pathway recalibration...*")
            success = bot.document_manager.regenerate_all_embeddings()
            if success:
                await status_message.edit(content="*neural pathways successfully recalibrated!* All document embeddings have been regenerated.")
            else:
                await status_message.edit(content="*neural pathway failure!* Failed to regenerate embeddings.")
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}")
            await interaction.followup.send(f"*neural circuit overload!* Error regenerating embeddings: {str(e)}")

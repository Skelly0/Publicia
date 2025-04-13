"""
Image handling commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import re
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from google.generativeai import types as genai_types
from PIL import Image
from io import BytesIO
import base64

from utils.helpers import split_message, is_image
from prompts.system_prompt import SYSTEM_PROMPT # Import system prompt for context if needed, though likely not directly used for image gen prompt

logger = logging.getLogger(__name__)

# --- Gemini Helper ---
def _get_gemini_client(config):
    """Initializes and returns a Gemini client."""
    try:
        # Use the GOOGLE_API_KEY from the bot's config
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in configuration.")
            return None
        # Configure the client directly with the API key
        genai.configure(api_key=api_key)
        # Create and return the client instance
        client = genai.GenerativeModel(model_name="gemini-2.0-flash-exp-image-generation") # Use GenerativeModel directly
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None

def register_commands(bot):
    """Register all image handling commands with the bot."""

    @bot.command(name="add_image", brief="Add an image to the knowledge base. \nUsage: `Publicia! add_image \"Your Image Name\" [yes/no]` \n(yes/no controls whether to auto-generate a description, default is yes)")
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
                description = await bot._generate_image_description(image_data)
                if description == "Error generating description.":
                    await ctx.send("*neural circuit overload!* An error occurred while processing the image.")
                    return
                description = name + ": " + description
                
                # Add to image manager
                image_id = await bot.image_manager.add_image(name, image_data, description)
                
                # Success message with preview of auto-generated description
                description_preview = description[:1000] + "..." if len(description) > 1000 else description
                success_message = f"*neural analysis complete!* Added image '{name}' to my knowledge base with ID: {image_id}\n\nGenerated description: {description_preview}"
                await status_msg.edit(content=success_message)
            else:
                # Ask user to provide a description
                await status_msg.edit(content="Please provide a description for the image (type it and send within 60 seconds):")
                
                try:
                    # Wait for user to type description
                    description_msg = await bot.wait_for(
                        'message',
                        timeout=60.0,
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel
                    )
                    description = description_msg.content
                    
                    # Add to image manager
                    image_id = await bot.image_manager.add_image(name, image_data, description)
                    
                    await ctx.send(f"*neural pathways reconfigured!* Added image '{name}' with your custom description to my knowledge base with ID: {image_id}")
                except asyncio.TimeoutError:
                    await status_msg.edit(content="*neural pathway timeout!* You didn't provide a description within the time limit.")
                    return
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await ctx.send("*neural circuit overload!* An error occurred while processing the image.")

    @bot.tree.command(name="list_images", description="List all images in Publicia's knowledge base")
    async def list_images(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            images = bot.image_manager.list_images()
            
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

    @bot.tree.command(name="view_image", description="View an image from Publicia's knowledge base")
    @app_commands.describe(image_id="ID of the image to view")
    async def view_image(interaction: discord.Interaction, image_id: str):
        await interaction.response.defer()
        try:
            # Check if image exists
            if image_id not in bot.image_manager.metadata:
                await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                return
            
            # Get image metadata
            image_meta = bot.image_manager.metadata[image_id]
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

    @bot.command(name="edit_image", brief="View and edit an image description. Usage: Publicia! edit_image [image_id]")
    async def edit_image_prefix(ctx, image_id: str):
        """View and edit an image description with a conversational flow."""
        try:
            # Check if image exists
            if image_id not in bot.image_manager.metadata:
                await ctx.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                return
            
            # Get image metadata
            image_meta = bot.image_manager.metadata[image_id]
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
                message = await bot.wait_for('message', timeout=60.0, check=check)
                
                # Check if user wants to cancel
                if message.content.lower() == 'cancel':
                    await ctx.send(f"*neural pathway unchanged!* Keeping the current description for image '{image_name}'.")
                    return
                
                # Update the description
                new_description = message.content
                success = await bot.image_manager.update_description(image_id, new_description)
                
                if success:
                    await ctx.send(f"*neural pathways reconfigured!* Updated description for image '{image_name}'.")
                else:
                    await ctx.send(f"*neural error detected!* Failed to update description for image '{image_name}'.")
            
            except asyncio.TimeoutError:
                await ctx.send("*neural pathway timeout!* No description provided within the time limit.")
                
        except Exception as e:
            logger.error(f"Error editing image description: {e}")
            await ctx.send("*neural circuit overload!* An error occurred while processing the image.")

    @bot.tree.command(name="remove_image", description="Remove an image from Publicia's knowledge base")
    @app_commands.describe(image_id="ID of the image to remove")
    async def remove_image(interaction: discord.Interaction, image_id: str):
        await interaction.response.defer()
        try:
            success = bot.image_manager.delete_image(image_id)
            
            if success:
                await interaction.followup.send(f"*neural pathways reconfigured!* Removed image with ID: {image_id}")
            else:
                await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                
        except Exception as e:
            logger.error(f"Error removing image: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while removing the image.")

    @bot.tree.command(name="update_image_description", description="Update the description for an image")
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

            if image_id not in bot.image_manager.metadata:
                await interaction.followup.send(f"*neural error detected!* Could not find image with ID: {image_id}")
                return

            success = await bot.image_manager.update_description(image_id, description) # Added await

            if success:
                await interaction.followup.send(f"*neural pathways reconfigured!* Updated description for image with ID: {image_id}")
            else:
                await interaction.followup.send(f"*neural error detected!* Could not update image description for ID: {image_id}")

        except Exception as e:
            logger.error(f"Error updating image description: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while updating the image description.")

    # --- Gemini Image Generation Command ---
    @bot.tree.command(name="generate_gemini_image", description="Generate an image using Gemini 2.0 Flash based on a text prompt.")
    @app_commands.describe(prompt="The text prompt to generate an image from.")
    async def generate_gemini_image_cmd(interaction: discord.Interaction, prompt: str):
        """Generates an image using Gemini based on the provided prompt, potentially using document context."""
        await interaction.response.defer()
        status_msg = await interaction.followup.send("*neural pathways activating... analyzing prompt and searching knowledge base...*")

        try:
            # --- Fetch Document Context ---
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id),
                default_model=bot.config.LLM_MODEL # Use default LLM for context search config
            )
            max_results = bot.config.get_top_k_for_model(preferred_model) # Use config for consistency

            logger.info(f"Fetching context for image generation prompt (max_results={max_results}): {prompt}")
            search_results = await bot.process_hybrid_query(
                prompt,
                interaction.user.name,
                max_results=max_results,
                use_context=False # Keep this False for slash commands
            )

            doc_context = ""
            if search_results:
                logger.info(f"Found {len(search_results)} relevant document sections for image prompt.")
                # Simple formatting for image prompt context
                context_chunks = [chunk for doc, chunk, score, image_id, chunk_index, total_chunks in search_results if not image_id] # Exclude image descriptions themselves
                if context_chunks:
                    doc_context = "Relevant Context:\n" + "\n\n".join(context_chunks)
                    doc_context = doc_context[:3000] # Limit context size for prompt
                    await status_msg.edit(content=f"*neural pathways activating... found {len(context_chunks)} relevant text sections... contacting Gemini for image generation...*")
                else:
                     await status_msg.edit(content="*neural pathways activating... no relevant text sections found... contacting Gemini for image generation...*")

            else:
                logger.info("No relevant document sections found for image prompt.")
                await status_msg.edit(content="*neural pathways activating... contacting Gemini for image generation...*")


            # --- Initialize Gemini Client ---
            gemini_client = _get_gemini_client(bot.config)
            if not gemini_client:
                await status_msg.edit(content="*neural circuit error!* Failed to initialize Gemini client. Check API key configuration.")
                return

            # --- Prepare Final Prompt ---
            final_prompt = f"{doc_context}\n\nUser Prompt: {prompt}" if doc_context else prompt
            logger.info(f"Generating Gemini image with final prompt:\n{final_prompt}")

            # Call Gemini API
            response = await gemini_client.generate_content_async( # Use async version
                contents=[final_prompt], # Use the combined prompt
                # generation_config=genai_types.GenerationConfig(
                #     response_modalities=['Text', 'Image'] # REMOVED: Causes error
                # )
                # No specific generation config needed for basic image generation here unless tuning parameters
            )

            generated_text = ""
            image_data = None

            # Process response parts
            for part in response.candidates[0].content.parts:
                if part.text:
                    generated_text += part.text + "\n"
                elif hasattr(part, 'inline_data') and part.inline_data: # Check for inline_data attribute
                    # Decode base64 image data
                    try:
                        image_bytes = base64.b64decode(part.inline_data.data)
                        # Validate if it's actually image data using PIL
                        try:
                            img_test = Image.open(BytesIO(image_bytes))
                            img_test.verify() # Verify image data integrity
                            image_data = image_bytes # Store the valid bytes
                            logger.info("Successfully decoded and validated Gemini image data.")
                        except (Image.UnidentifiedImageError, SyntaxError, IOError) as img_err:
                            logger.error(f"Gemini returned invalid image data: {img_err}")
                            # Optionally log the first few bytes of the data for debugging
                            logger.debug(f"Invalid image data (first 100 bytes): {part.inline_data.data[:100]}")
                    except base64.binascii.Error as b64_err:
                        logger.error(f"Failed to decode base64 image data from Gemini: {b64_err}")
                    except Exception as decode_err:
                         logger.error(f"An unexpected error occurred during image decoding: {decode_err}")


            if not image_data:
                await status_msg.edit(content=f"*neural pathway incomplete!* Gemini did not return a valid image.\n\n**Gemini's Text Response:**\n{generated_text if generated_text else 'No text response.'}")
                return

            # Save the image using ImageManager
            await status_msg.edit(content="*neural pathways stabilizing... saving generated image...*")
            image_name = f"gemini_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            context_note = "(Used document context)" if doc_context else "(No document context used)"
            description = f"Generated by Gemini {context_note} with prompt: '{prompt}'\n\nGemini Text: {generated_text if generated_text else 'None'}"

            image_id = await bot.image_manager.add_image(image_name, image_data, description)

            # Send the result
            image_path = Path(bot.image_manager.metadata[image_id]['path'])
            with open(image_path, 'rb') as f:
                file = discord.File(f, filename=f"{image_name}.png")
                context_used_msg = " using relevant document context" if doc_context else ""
                response_text = f"*neural synthesis complete!* Generated image (ID: {image_id}){context_used_msg} based on prompt: '{prompt}'\n\n**Gemini's Text Response:**\n{generated_text if generated_text else 'No text response.'}"
                # Split message if needed
                for chunk in split_message(response_text):
                     await interaction.followup.send(chunk) # Send text chunks first
                await interaction.followup.send(file=file) # Then send the file

            # Clean up original status message if possible and needed (followup sends new messages)
            try:
                await status_msg.delete()
            except discord.NotFound:
                pass # Ignore if already deleted
            except Exception as del_err:
                logger.warning(f"Could not delete original status message: {del_err}")


        except Exception as e:
            logger.error(f"Error generating Gemini image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await status_msg.edit(content=f"*neural circuit overload!* An error occurred during Gemini image generation: {e}")
            except discord.NotFound:
                 await interaction.followup.send(f"*neural circuit overload!* An error occurred during Gemini image generation: {e}")
            except Exception as final_err:
                 logger.error(f"Failed to send error message: {final_err}")


    # --- Gemini Image Editing Command ---
    @bot.tree.command(name="edit_gemini_image", description="Edit an existing image using Gemini 2.0 Flash based on a text prompt.")
    @app_commands.describe(
        image_id="The ID of the image in Publicia's knowledge base to edit.",
        prompt="The text prompt describing the edits."
    )
    async def edit_gemini_image_cmd(interaction: discord.Interaction, image_id: str, prompt: str):
        """Edits an existing image using Gemini based on the provided prompt and image ID."""
        await interaction.response.defer()
        status_msg = await interaction.followup.send("*neural pathways activating... retrieving image and contacting Gemini for editing...*")

        try:
            # 1. Retrieve the image from ImageManager
            if image_id not in bot.image_manager.metadata:
                await status_msg.edit(content=f"*neural error detected!* Could not find image with ID: {image_id}")
                return

            try:
                input_image_data, _ = bot.image_manager.get_image(image_id)
                # Validate input image data
                try:
                    img_test = Image.open(BytesIO(input_image_data))
                    img_test.verify()
                except (Image.UnidentifiedImageError, SyntaxError, IOError) as img_err:
                    logger.error(f"Input image {image_id} is corrupted or invalid: {img_err}")
                    await status_msg.edit(content=f"*neural error detected!* The input image (ID: {image_id}) appears to be corrupted or invalid.")
                    return
            except FileNotFoundError:
                await status_msg.edit(content=f"*neural error detected!* Image file not found for ID: {image_id}")
                return
            except Exception as retrieve_err:
                 logger.error(f"Error retrieving image {image_id}: {retrieve_err}")
                 await status_msg.edit(content=f"*neural error detected!* Failed to retrieve input image (ID: {image_id}).")
                 return

            # 2. Initialize Gemini Client
            gemini_client = _get_gemini_client(bot.config)
            if not gemini_client:
                await status_msg.edit(content="*neural circuit error!* Failed to initialize Gemini client. Check API key configuration.")
                return

            logger.info(f"Editing Gemini image {image_id} with prompt: {prompt}")

            # 3. Prepare API Contents (Prompt + Image)
            # Determine MIME type (assuming PNG as saved by ImageManager)
            mime_type = "image/png"
            contents = [
                prompt, # Text prompt first
                # Then the image data
                {'mime_type': mime_type, 'data': base64.b64encode(input_image_data).decode('utf-8')}
            ]

            # 4. Call Gemini API
            response = await gemini_client.generate_content_async( # Use async version
                contents=contents,
                generation_config=genai_types.GenerationConfig(
                    response_modalities=['Text', 'Image']
                )
            )

            generated_text = ""
            edited_image_data = None

            # 5. Process response parts
            for part in response.candidates[0].content.parts:
                if part.text:
                    generated_text += part.text + "\n"
                elif hasattr(part, 'inline_data') and part.inline_data:
                     # Decode base64 image data
                    try:
                        image_bytes = base64.b64decode(part.inline_data.data)
                        # Validate if it's actually image data using PIL
                        try:
                            img_test = Image.open(BytesIO(image_bytes))
                            img_test.verify() # Verify image data integrity
                            edited_image_data = image_bytes # Store the valid bytes
                            logger.info("Successfully decoded and validated Gemini edited image data.")
                        except (Image.UnidentifiedImageError, SyntaxError, IOError) as img_err:
                            logger.error(f"Gemini returned invalid edited image data: {img_err}")
                            logger.debug(f"Invalid edited image data (first 100 bytes): {part.inline_data.data[:100]}")
                    except base64.binascii.Error as b64_err:
                        logger.error(f"Failed to decode base64 edited image data from Gemini: {b64_err}")
                    except Exception as decode_err:
                         logger.error(f"An unexpected error occurred during edited image decoding: {decode_err}")


            if not edited_image_data:
                await status_msg.edit(content=f"*neural pathway incomplete!* Gemini did not return a valid edited image.\n\n**Gemini's Text Response:**\n{generated_text if generated_text else 'No text response.'}")
                return

            # 6. Save the *new* edited image using ImageManager
            await status_msg.edit(content="*neural pathways stabilizing... saving edited image...*")
            original_name = bot.image_manager.metadata[image_id]['name']
            new_image_name = f"gemini_edit_of_{original_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = f"Edited by Gemini from image ID '{image_id}' with prompt: '{prompt}'\n\nGemini Text: {generated_text if generated_text else 'None'}"

            new_image_id = await bot.image_manager.add_image(new_image_name, edited_image_data, description)

            # 7. Send the result
            new_image_path = Path(bot.image_manager.metadata[new_image_id]['path'])
            with open(new_image_path, 'rb') as f:
                file = discord.File(f, filename=f"{new_image_name}.png")
                response_text = f"*neural synthesis complete!* Edited image (Original ID: {image_id}, New ID: {new_image_id}) based on prompt: '{prompt}'\n\n**Gemini's Text Response:**\n{generated_text if generated_text else 'No text response.'}"
                 # Split message if needed
                for chunk in split_message(response_text):
                     await interaction.followup.send(chunk) # Send text chunks first
                await interaction.followup.send(file=file) # Then send the file

            # Clean up original status message
            try:
                await status_msg.delete()
            except discord.NotFound:
                pass
            except Exception as del_err:
                logger.warning(f"Could not delete original status message: {del_err}")

        except Exception as e:
            logger.error(f"Error editing Gemini image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await status_msg.edit(content=f"*neural circuit overload!* An error occurred during Gemini image editing: {e}")
            except discord.NotFound:
                 await interaction.followup.send(f"*neural circuit overload!* An error occurred during Gemini image editing: {e}")
            except Exception as final_err:
                 logger.error(f"Failed to send error message: {final_err}")

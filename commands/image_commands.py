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
from utils.helpers import split_message, is_image

logger = logging.getLogger(__name__)

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

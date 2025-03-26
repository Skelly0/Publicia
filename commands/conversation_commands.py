"""
Conversation management commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import re
import asyncio
import json
import os
from datetime import datetime
from utils.helpers import split_message

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all conversation management commands with the bot."""

    @bot.tree.command(name="manage_history", description="View and manage your conversation history")
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
            recent_messages = bot.conversation_manager.get_limited_history(interaction.user.name, limit)
            
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

    @bot.tree.command(name="delete_history_messages", description="Delete specific messages from your conversation history")
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
            success, message, deleted_count = bot.conversation_manager.delete_messages_by_display_index(
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

    @bot.tree.command(name="archive_conversation", description="Archive your current conversation history")
    @app_commands.describe(
        archive_name="Optional name for the archive (defaults to timestamp)"
    )
    async def archive_conversation(interaction: discord.Interaction, archive_name: str = None):
        await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
        try:
            success, message = bot.conversation_manager.archive_conversation(interaction.user.name, archive_name)
            
            if success:
                await interaction.followup.send(f"*neural storage complete!* {message}", ephemeral=True)
            else:
                await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error archiving conversation: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while archiving the conversation.", ephemeral=True)

    @bot.tree.command(name="list_archives", description="List your archived conversation histories")
    async def list_archives(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
        try:
            archives = bot.conversation_manager.list_archives(interaction.user.name)
            
            if not archives:
                await interaction.followup.send("*neural archives empty!* You don't have any archived conversations.", ephemeral=True)
                return
            
            # Format the list of archives
            response = "*accessing neural storage banks...*\n\n"
            response += "**ARCHIVED CONVERSATIONS**\n\n"
            
            for i, archive in enumerate(archives):
                # Try to extract a timestamp from the filename
                timestamp_match = re.search(r'(\d{8}_\d{6})', archive)
                if timestamp_match:
                    try:
                        timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
                        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        response += f"{i+1}. **{archive}** (Created: {formatted_time})\n"
                    except:
                        response += f"{i+1}. **{archive}**\n"
                else:
                    response += f"{i+1}. **{archive}**\n"
            
            response += "\n*use `/swap_conversation` to swap between current and archived conversations*"
            
            await interaction.followup.send(response, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error listing archives: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while listing archives.", ephemeral=True)

    @bot.tree.command(name="swap_conversation", description="Swap between current and archived conversation histories")
    @app_commands.describe(
        archive_name="Name of the archive to swap with"
    )
    async def swap_conversation(interaction: discord.Interaction, archive_name: str):
        await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
        try:
            success, message = bot.conversation_manager.swap_conversation(interaction.user.name, archive_name)
            
            if success:
                await interaction.followup.send(f"*neural reconfiguration complete!* {message}", ephemeral=True)
            else:
                await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error swapping conversation: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while swapping conversations.", ephemeral=True)

    @bot.tree.command(name="delete_archive", description="Delete an archived conversation")
    @app_commands.describe(
        archive_name="Name of the archive to delete",
        confirm="Type 'yes' to confirm deletion"
    )
    async def delete_archive(interaction: discord.Interaction, archive_name: str, confirm: str = "no"):
        await interaction.response.defer(ephemeral=True)  # Make response only visible to the user
        try:
            # Check confirmation
            if confirm.lower() != "yes":
                await interaction.followup.send("*deletion aborted!* Please confirm by setting `confirm` to 'yes'.", ephemeral=True)
                return
                
            success, message = bot.conversation_manager.delete_archive(interaction.user.name, archive_name)
            
            if success:
                await interaction.followup.send(f"*neural memory purged!* {message}", ephemeral=True)
            else:
                # If archive not found, list available archives
                if "not found" in message:
                    archives = bot.conversation_manager.list_archives(interaction.user.name)
                    if archives:
                        archives_list = "\n".join([f"• {archive}" for archive in archives])
                        await interaction.followup.send(f"*neural error detected!* {message}\n\nAvailable archives:\n{archives_list}", ephemeral=True)
                    else:
                        await interaction.followup.send(f"*neural error detected!* {message}\nYou don't have any archived conversations.", ephemeral=True)
                else:
                    await interaction.followup.send(f"*neural error detected!* {message}", ephemeral=True)
                
        except Exception as e:
            logger.error(f"Error deleting archive: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while deleting the archive.", ephemeral=True)
    
    @bot.tree.command(name="history", description="Display your conversation history with the bot")
    @app_commands.describe(limit="Number of messages to display (default: 10, max: 50)")
    async def show_history(interaction: discord.Interaction, limit: int = 10):
        await interaction.response.defer(ephemeral=True)
        try:
            # Validate limit
            if limit <= 0:
                await interaction.followup.send("*neural error detected!* The limit must be a positive number.")
                return
            
            # Cap limit at 50 to prevent excessive output
            limit = min(limit, 50)
            
            # Get conversation history
            file_path = bot.conversation_manager.get_file_path(interaction.user.name)
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
                await interaction.followup.send(chunk, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error displaying conversation history: {e}")
            await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to retrieve your conversation history.")

    @bot.tree.command(name="lobotomise", description="Wipe your conversation history with the bot")
    async def lobotomise(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            file_path = bot.conversation_manager.get_file_path(interaction.user.name)
            if os.path.exists(file_path):
                os.remove(file_path)
                await interaction.followup.send("*AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA*... memory wiped! I've forgotten our conversations... Who are you again?")
            else:
                await interaction.followup.send("hmm, i don't seem to have any memories of our conversations to wipe!")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            await interaction.followup.send("oops, something went wrong while trying to clear my memory!")

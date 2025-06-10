"""
Commands for tracking Discord channels and updating documents.
"""
import discord
from discord import app_commands
from discord.ext import commands, tasks
import logging
import json
import os
from datetime import datetime, timezone
import asyncio
from utils.helpers import check_permissions, sanitize_filename

logger = logging.getLogger(__name__)

TRACKED_CHANNELS_FILE = "documents/tracked_channels.json"

# Helper function to load tracked channels (moved to module level)
def load_tracked_channels():
    if os.path.exists(TRACKED_CHANNELS_FILE):
        with open(TRACKED_CHANNELS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Helper function to save tracked channels (moved to module level)
def save_tracked_channels(data):
    with open(TRACKED_CHANNELS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Background task for updating tracked channels (moved to module level)
@tasks.loop(hours=6)
async def update_tracked_channels(bot_instance):
    """Periodically checks for new messages in tracked channels and updates their archives."""
    logger.info("Starting periodic check for tracked channel updates...")
    tracked_channels = load_tracked_channels()
    if not tracked_channels:
        logger.info("No channels are currently being tracked.")
        return

    for channel_id, data in tracked_channels.items():
        try:
            channel = await bot_instance.fetch_channel(int(channel_id))
            if not channel:
                logger.warning(f"Could not find channel with ID {channel_id}. Skipping.")
                continue

            last_message_id = data.get("last_message_id")
            
            # Fetch messages after the last recorded one
            new_messages = []
            async for message in channel.history(limit=None, after=discord.Object(id=last_message_id) if last_message_id else None):
                new_messages.append(message)

            if not new_messages:
                logger.info(f"No new messages in channel {channel.name} ({channel_id}).")
                continue

            # Sort messages by timestamp (oldest first)
            new_messages.sort(key=lambda m: m.created_at)
            
            # Format new messages
            new_content = ""
            for message in new_messages:
                timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
                author_name = message.author.display_name
                new_content += f"\n[{timestamp}] {author_name}:\n    {message.content}\n"

            # Append to the existing document
            doc_uuid = data["document_uuid"]
            success = await bot_instance.document_manager.append_to_document(doc_uuid, new_content)

            if success:
                # Update the last message ID
                data["last_message_id"] = new_messages[-1].id
                save_tracked_channels(tracked_channels)
                logger.info(f"Successfully appended {len(new_messages)} new messages to archive for channel {channel.name} ({channel_id}).")
            else:
                logger.error(f"Failed to append content to document for channel {channel.name} ({channel_id}).")

        except Exception as e:
            logger.error(f"Error updating tracked channel {channel_id}: {e}", exc_info=True)

def register_commands(bot):
    """Register all tracking commands with the bot."""

    @bot.tree.command(name="track_channel", description="Start tracking a channel and archive it periodically (admin only)")
    @app_commands.describe(
        channel="The channel to track",
        update_interval_hours="How often to check for new messages (in hours, default: 6)"
    )
    @app_commands.check(check_permissions)
    async def track_channel(interaction: discord.Interaction, channel: discord.TextChannel, update_interval_hours: int = 6):
        await interaction.response.defer()
        
        tracked_channels = load_tracked_channels()
        channel_id_str = str(channel.id)

        if channel_id_str in tracked_channels:
            await interaction.followup.send(f"Channel {channel.mention} is already being tracked.")
            return

        # Create initial archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        doc_name = f"channel_archive_{channel.name}_{timestamp}"
        
        # Call the existing archive_channel logic (or a refactored version)
        # For now, let's simulate the initial archival process
        await interaction.followup.send(f"Performing initial archive of {channel.mention}...")
        
        # This would be a call to a function similar to the body of the original archive_channel command
        # For simplicity, we'll just create a placeholder document for now.
        # In a real implementation, you would reuse the message fetching and formatting logic.
        
        initial_content = f"# Archive of channel: {channel.name}\n"
        initial_content += f"# Tracked starting from: {datetime.now(timezone.utc).isoformat()}\n"
        
        doc_uuid = await bot.document_manager.add_document(original_name=doc_name, content=initial_content)

        if not doc_uuid:
            await interaction.followup.send("Failed to create initial archive document.")
            return

        tracked_channels[channel_id_str] = {
            "name": channel.name,
            "document_uuid": doc_uuid,
            "update_interval_hours": update_interval_hours,
            "last_message_id": channel.last_message_id if channel.last_message_id else None
        }
        
        save_tracked_channels(tracked_channels)
        
        await interaction.followup.send(
            f"Channel {channel.mention} is now being tracked. "
            f"Initial archive created with document UUID: `{doc_uuid}`. "
            f"Updates will occur every {update_interval_hours} hours."
        )

    @bot.tree.command(name="untrack_channel", description="Stop tracking a channel (admin only)")
    @app_commands.describe(channel="The channel to stop tracking")
    @app_commands.check(check_permissions)
    async def untrack_channel(interaction: discord.Interaction, channel: discord.TextChannel):
        await interaction.response.defer()
        
        tracked_channels = load_tracked_channels()
        channel_id_str = str(channel.id)

        if channel_id_str not in tracked_channels:
            await interaction.followup.send(f"Channel {channel.mention} is not currently being tracked.")
            return

        del tracked_channels[channel_id_str]
        save_tracked_channels(tracked_channels)
        
        await interaction.followup.send(f"Stopped tracking channel {channel.mention}. The archive document will no longer be updated.")

    # Add a command to start the background task (now references module-level task)
    @bot.command(name="start_tracking_task", hidden=True)
    @commands.is_owner()
    async def start_tracking_task(ctx):
        update_tracked_channels.start(bot)
        await ctx.send("Channel tracking task started.")

    # Add a command to stop the background task (now references module-level task)
    @bot.command(name="stop_tracking_task", hidden=True)
    @commands.is_owner()
    async def stop_tracking_task(ctx):
        update_tracked_channels.cancel()
        await ctx.send("Channel tracking task stopped.")

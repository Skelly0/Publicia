"""
Utility functions for Publicia bot
"""
import re
import discord
import string # Added for filename sanitization
from typing import List
from discord import app_commands
from managers.config import Config  # Import the Config class

# Instantiate Config to access settings
config = Config()


_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")   # ASCII 0-31

def sanitize_filename(filename: str) -> str:
    if not filename:
        return "_unnamed_file_"

    # drop real control characters
    filename = _CONTROL_CHARS.sub("", filename)

    # replace Windows-forbidden punctuation
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # collapse multiple underscores
    filename = re.sub(r"_+", "_", filename)

    # trim leading/trailing spaces & underscores
    filename = filename.strip(" _")

    if not filename:
        return "_sanitized_empty_"

    # dodge Windows reserved names
    base, ext = os.path.splitext(filename)
    if base.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }:
        filename = f"_{base}{ext}"

    # keep it (reasonably) short
    MAX_LEN = 100
    if len(filename) > MAX_LEN:
        base, ext = os.path.splitext(filename)
        filename = f"{base[:MAX_LEN - len(ext)]}{ext}"

    return filename


async def check_permissions(ctx):
    """
    Check if a user has permissions for a command, compatible with both
    regular commands (discord.ext.commands.Context) and slash commands
    (discord.Interaction).
    """
    is_interaction = isinstance(ctx, discord.Interaction)

    # Determine user and guild from the context type
    user = ctx.user if is_interaction else ctx.author
    guild = ctx.guild

    # 1. Check for the override user ID
    if user.id == 203229662967627777:
        return True

    # 2. Check if the user is in the allowed list
    if user.id in config.ALLOWED_USER_IDS:
        return True

    # 3. Check for allowed roles if in a guild
    if guild and config.ALLOWED_ROLE_IDS:
        # Ensure we have a member object to check roles
        member = guild.get_member(user.id)
        if member:
            user_role_ids = {role.id for role in member.roles}
            if user_role_ids.intersection(set(config.ALLOWED_ROLE_IDS)):
                return True

    # 4. If all checks fail, deny permission
    error_message = "You do not have the required permissions to use this command."

    # For slash commands, send an ephemeral message and raise a specific exception
    if is_interaction:
        try:
            if not ctx.response.is_done():
                await ctx.response.send_message(error_message, ephemeral=True)
            else:
                await ctx.followup.send(error_message, ephemeral=True)
        except discord.HTTPException as e:
            print(f"Failed to send permission error to user {user.id}: {e}")
        # Raising CheckFailure is the standard way to halt a slash command
        raise app_commands.CheckFailure(error_message)
    else:
        # For prefix commands, returning False is sufficient to block execution
        # The bot's default error handler can notify the user if needed
        return False


def is_image(attachment):
    """Check if an attachment is an image based on content type or file extension."""
    if attachment.content_type and attachment.content_type.startswith('image/'):
        return True
    # Fallback to checking file extension
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
    return any(attachment.filename.lower().endswith(ext) for ext in image_extensions)


def split_message(text, max_length=1750):
    """smarter message splitting that respects semantic boundaries"""
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    chunks = []
    current_chunk = ""
    
    # try paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk + ('\n\n' if current_chunk else '') + paragraph) <= max_length:
            current_chunk += ('\n\n' if current_chunk else '') + paragraph
        else:
            # store current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            
            # if paragraph itself is too long
            if len(paragraph) > max_length:
                # try line-by-line
                lines = paragraph.split('\n')
                current_chunk = ""
                
                for line in lines:
                    if len(current_chunk + ('\n' if current_chunk else '') + line) <= max_length:
                        current_chunk += ('\n' if current_chunk else '') + line
                    else:
                        # if line is too long
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        
                        # smart splitting for long lines
                        if len(line) > max_length:
                            # try splitting at these boundaries in order
                            split_markers = ['. ', '? ', '! ', '; ', ', ', ' - ', ' ']
                            
                            start = 0
                            while start < len(line):
                                # find best split point
                                end = start + max_length
                                if end >= len(line):
                                    chunk = line[start:]
                                    if current_chunk and len(current_chunk + chunk) <= max_length:
                                        current_chunk += chunk
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        current_chunk = chunk
                                    break
                                
                                # try each split marker
                                split_point = end
                                for marker in split_markers:
                                    pos = line[start:end].rfind(marker)
                                    if pos > 0:  # found a good split point
                                        split_point = start + pos + len(marker)
                                        break
                                
                                chunk = line[start:split_point]
                                if current_chunk and len(current_chunk + chunk) <= max_length:
                                    current_chunk += chunk
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = chunk
                                
                                start = split_point
                        else:
                            current_chunk = line
            else:
                current_chunk = paragraph
    
    # add final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


# --- Add import for os used in sanitize_filename ---
import os

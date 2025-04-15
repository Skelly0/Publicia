"""
Utility functions for Publicia bot
"""
import re
import discord
from typing import List
from discord import app_commands
from managers.config import Config  # Import the Config class

# Instantiate Config to access settings
config = Config()

async def check_permissions(interaction: discord.Interaction):
    """Check if a user has permissions based on configured user IDs or role IDs."""
    user_id = interaction.user.id

    # 1. Check the special override user ID
    if user_id == 203229662967627777:
        return True

    # 2. Check if the user ID is in the allowed list from config
    if user_id in config.ALLOWED_USER_IDS:
        return True

    # 3. Check roles only if in a guild and allowed roles are configured
    if interaction.guild and config.ALLOWED_ROLE_IDS:
        try:
            # Ensure we have the member object, fetching if necessary
            member = interaction.user
            if not isinstance(member, discord.Member): # If interaction.user is just a User object
                 member = interaction.guild.get_member(user_id)
                 if member is None:
                     member = await interaction.guild.fetch_member(user_id)

            if member: # Proceed only if member object is available
                user_role_ids = {role.id for role in member.roles}
                allowed_role_ids_set = set(config.ALLOWED_ROLE_IDS)

                # Check for intersection between user's roles and allowed roles
                if user_role_ids.intersection(allowed_role_ids_set):
                    return True

        except discord.NotFound:
            print(f"Could not find member {user_id} in guild {interaction.guild.id} for permission check.")
            return False # User not found in guild
        except discord.Forbidden:
            print(f"Bot lacks permissions to fetch member {user_id} or roles in guild {interaction.guild.id}.")
            return False # Bot permission issue
        except Exception as e:
            print(f"Error checking roles for user {user_id} in guild {interaction.guild.id}: {e}")
            return False # General error during role check

    # 4. If none of the above checks passed, deny permission
    # Raise CheckFailure to give specific feedback in Discord
    raise app_commands.CheckFailure(
        "You do not have the required permissions (specific user ID or allowed role) to use this command."
    )


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

"""
Utility functions for Publicia bot
"""
import re
import discord
from typing import List
from discord import app_commands

async def check_permissions(interaction: discord.Interaction):
    """Check if a user has administrator permissions."""
    # First check for special user ID (this doesn't require guild permissions)
    if interaction.user.id == 203229662967627777:
        return True
        
    # Check if we're in a guild
    if not interaction.guild:
        raise app_commands.CheckFailure("This command can only be used in a server")
    
    # Try to get permissions directly from interaction.user
    try:
        return interaction.user.guild_permissions.administrator
    except AttributeError:
        # If that fails, try getting the member object
        try:
            member = interaction.guild.get_member(interaction.user.id)
            
            # Member might be None if not in cache
            if member is None:
                # Fetch fresh from API
                member = await interaction.guild.fetch_member(interaction.user.id)
                
            return member.guild_permissions.administrator
        except Exception as e:
            print(f"Permission check error: {e}")
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

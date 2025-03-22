"""
Command registration for Publicia Discord bot.
Import all command modules here for easy access.
"""

from . import document_commands
from . import image_commands
from . import conversation_commands
from . import admin_commands
from . import utility_commands
from . import query_commands

__all__ = [
    'document_commands',
    'image_commands',
    'conversation_commands',
    'admin_commands',
    'utility_commands',
    'query_commands'
]

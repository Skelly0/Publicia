"""Prompt templates for Publicia."""
from .system_prompt import SYSTEM_PROMPT, INFORMATIONAL_SYSTEM_PROMPT, get_system_prompt_with_documents, get_informational_system_prompt_with_documents
from .image_prompt import IMAGE_DESCRIPTION_PROMPT

__all__ = [
    'SYSTEM_PROMPT',
    'INFORMATIONAL_SYSTEM_PROMPT',
    'get_system_prompt_with_documents',
    'get_informational_system_prompt_with_documents',
    'IMAGE_DESCRIPTION_PROMPT'
]

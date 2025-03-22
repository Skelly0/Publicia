#!/usr/bin/env python3
"""
Publicia Discord Bot - Entry Point
A Discord bot for Ledus Banum 77 and Imperial Lore QA
"""
import asyncio
import sys
from utils.logging import configure_logging, display_startup_banner
from managers.config import Config
from managers.documents import DocumentManager
from managers.images import ImageManager
from managers.conversation import ConversationManager
from managers.preferences import UserPreferencesManager
from bot import DiscordBot

logger = configure_logging()

async def main():
    """Main entry point for the Publicia Discord bot."""
    try:
        # Display startup banner
        display_startup_banner()
        
        # Initialize configuration
        config = Config()
        
        # Initialize document manager
        document_manager = DocumentManager(top_k=config.TOP_K, config=config)
        
        # Clean up any empty documents
        empty_docs = document_manager.cleanup_empty_documents()
        if empty_docs:
            logger.info(f"Cleaned up {len(empty_docs)} empty documents at startup")
        
        # Initialize image manager with reference to document manager
        image_manager = ImageManager(document_manager=document_manager)
        
        # Initialize conversation manager
        conversation_manager = ConversationManager()
        
        # Initialize user preferences
        user_preferences_manager = UserPreferencesManager()
        
        # Create and start the bot
        bot = DiscordBot(
            config=config,
            document_manager=document_manager,
            image_manager=image_manager,
            conversation_manager=conversation_manager,
            user_preferences_manager=user_preferences_manager
        )
        
        async with bot:
            await bot.start(config.DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

"""
Utility commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import json
import os
from datetime import datetime
from utils.helpers import split_message
from prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all utility commands with the bot."""

    @bot.tree.command(name="list_commands", description="List all available commands")
    async def list_commands(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            response = "*accessing command database through enhanced synapses...*\n\n"
            response += "**AVAILABLE COMMANDS**\n\n"
            
            # SLASH COMMANDS SECTION
            response += "**Slash Commands** (`/command`)\n\n"
            
            categories = {
                "Lore Queries": ["query"],
                "Document Management": ["add_info", "list_docs", "remove_doc", "search_docs", "add_googledoc", "list_googledocs", "remove_googledoc", "rename_document", "list_files", "retrieve_file", "archive_channel"],
                "Image Management": ["list_images", "view_image", "edit_image", "remove_image", "update_image_description"],
                "Utility": ["list_commands", "set_model", "get_model", "toggle_debug", "help", "export_prompt", "reload_docs"],
                "Memory Management": ["lobotomise", "history", "manage_history", "delete_history_messages", "parse_channel", "archive_conversation", "list_archives", "swap_conversation", "delete_archive"], 
                "Moderation": ["ban_user", "unban_user"]
            }
            
            for category, cmd_list in categories.items():
                response += f"__*{category}*__\n"
                for cmd_name in cmd_list:
                    cmd = bot.tree.get_command(cmd_name)
                    if cmd:
                        desc = cmd.description or "No description available"
                        response += f"`/{cmd_name}`: {desc}\n"
                response += "\n"
            
            # PREFIX COMMANDS SECTION
            response += "**Prefix Commands** (`Publicia! command`)\n\n"
            
            # Get prefix commands from the bot
            prefix_commands = sorted(bot.commands, key=lambda x: x.name)
            
            # Group prefix commands by category (estimate categories based on names)
            prefix_categories = {
                "Document Management": [],
                "Image Management": []
            }
            
            # Sort commands into categories
            for cmd in prefix_commands:
                if "doc" in cmd.name.lower():
                    prefix_categories["Document Management"].append(cmd)
                elif "image" in cmd.name.lower():
                    prefix_categories["Image Management"].append(cmd)
            
            # Format and add each category of prefix commands
            for category, cmds in prefix_categories.items():
                if cmds:  # Only show categories that have commands
                    response += f"__*{category}*__\n"
                    for cmd in cmds:
                        brief = cmd.brief or "No description available"
                        response += f"`Publicia! {cmd.name}`: {brief}\n"
                    response += "\n"
            
            response += "\n*you can ask questions about ledus banum 77 and imperial lore by pinging/mentioning me or using the /query command!*"
            response += "\n\n*my genetically enhanced brain is always ready to help... just ask!*"
            response += "\n\n*for a detailed guide on all my features, use the `/help` command!*"
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
        except Exception as e:
            logger.error(f"Error listing commands: {e}")
            await interaction.followup.send("*my enhanced neurons misfired!* couldn't retrieve command list right now...")

    @bot.tree.command(name="set_model", description="Set your preferred AI model for responses")
    @app_commands.describe(model="Choose the AI model you prefer")
    @app_commands.choices(model=[
        app_commands.Choice(name="Qwen QwQ 32B", value="qwen/qwq-32b:free"),
        app_commands.Choice(name="Gemini 2.0 Flash", value="google/gemini-2.0-flash-001"),
        app_commands.Choice(name="DeepSeek Chat V3 0324", value="deepseek/deepseek-chat-v3-0324:free"),
        app_commands.Choice(name="DeepSeek-R1", value="deepseek/deepseek-r1:free"),
        app_commands.Choice(name="Nous: Hermes 405B", value="nousresearch/hermes-3-llama-3.1-405b"),
        app_commands.Choice(name="Claude 3.5 Haiku", value="anthropic/claude-3.5-haiku:beta"),
        app_commands.Choice(name="Claude 3.5 Sonnet", value="anthropic/claude-3.5-sonnet:beta"),
        app_commands.Choice(name="Claude 3.7 Sonnet", value="anthropic/claude-3.7-sonnet:beta"),
        app_commands.Choice(name="Testing Model", value="eva-unit-01/eva-qwen-2.5-72b"),
        app_commands.Choice(name="Wayfarer 70B", value="latitudegames/wayfarer-large-70b-llama-3.3"),
        app_commands.Choice(name="Anubis Pro 105B", value="thedrummer/anubis-pro-105b-v1"),
        #app_commands.Choice(name="Phi-4 Multimodal", value="microsoft/phi-4-multimodal-instruct"),
    ])
    async def set_model(interaction: discord.Interaction, model: str):
        await interaction.response.defer()
        try:
            # Check if user is allowed to use Claude 3.7 Sonnet
            if model == "anthropic/claude-3.7-sonnet:beta" and str(interaction.user.id) != "203229662967627777" and not (interaction.guild and interaction.user.guild_permissions.administrator):
                await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                return

            if model == "anthropic/claude-3.5-sonnet:beta" and str(interaction.user.id) != "203229662967627777" and not (interaction.guild and interaction.user.guild_permissions.administrator):
                await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                return
                
            success = bot.user_preferences_manager.set_preferred_model(str(interaction.user.id), model)
            
            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            if "deepseek/deepseek-chat-v3-0324" in model:
                model_name = "DeepSeek Chat v3"
            elif "deepseek/deepseek-chat-v3" in model:
                model_name = "DeepSeek V3 0324"
            elif "deepseek/deepseek-r1" in model:
                model_name = "DeepSeek-R1"
            elif model.startswith("google/"):
                model_name = "Gemini 2.0 Flash"
            elif model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-3.5-sonnet" in model:
                model_name = "Claude 3.5 Sonnet"
            elif "claude-3.7-sonnet" in model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in model:
                model_name = "Qwen QwQ 32B"
            elif "unslopnemo" in model or "eva-unit-01/eva-qwen-2.5-72b" in model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in model:
                model_name = "Anubis Pro 105B"
            elif "microsoft/phi-4-multimodal-instruct" in model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in model:
                model_name = "Phi-3.5 Mini"
            
            if success:
                # Create a description of all model strengths
                # Create a description of all model strengths
                model_descriptions = [
                    f"**Qwen QwQ 32B**: RECOMMENDED - Great for roleplaying with strong lore accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Prone to minor hallucinations due to it being a small model, and it sometimes slips in Chinese phrases. Uses ({bot.config.get_top_k_for_model('qwen/qwq-32b:free')}) with the free model, otherwise uses ({bot.config.get_top_k_for_model('qwen/qwq-32b')}).",
                    f"**Gemini 2.0 Flash**: RECOMMENDED - Better for accurate citations, factual responses, document analysis, image viewing capabilities, and has very fast response times. Uses more search results ({bot.config.get_top_k_for_model('google/gemini-2.0-flash-001')}) for broader context.",
                    f"**DeepSeek Chat V3 0324**: Great for roleplaying, creative responses, and in-character immersion, but may make things up due to its creativity. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324:free')}) search results with the free version, otherwise uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324')}).",
                    f"**DeepSeek-R1**: Similar to V3, but with reasoning. Great for roleplaying, more creative responses, and in-character immersion, but sometimes may make things up due to its creativity (less than V3 due to its reasoning). With free version uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1:free')}) search results, otherwise uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1')}).",
                    f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({bot.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}) for balanced context.",
                    f"**Claude 3.5 Haiku**: Excellent for comprehensive lore analysis and nuanced understanding with creativity, and has image viewing capabilities. Also great for longer roleplays. Uses a moderate number of search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}) for balanced context.",
                    f"**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-sonnet')}) to save money.",
                    f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                    f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({bot.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                    f"**Wayfarer 70B**: Optimized for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({bot.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                    f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({bot.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                    #f"**Phi-4 Multimodal**: Microsoft's latest multimodal model with vision capabilities. It's not a good model. Uses ({bot.config.get_top_k_for_model('microsoft/phi-4-multimodal-instruct')}) search results.",
                ]
                
                response = f"*neural architecture reconfigured!* Your preferred model has been set to **{model_name}**.\n\n**Model strengths:**\n"
                response += "\n".join(model_descriptions)
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send("*synaptic error detected!* Failed to set your preferred model. Please try again later.")
                
        except Exception as e:
            logger.error(f"Error setting preferred model: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while setting your preferred model.")


    @bot.tree.command(name="get_model", description="Show your currently selected AI model and available models")
    async def get_model(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id), 
                default_model=bot.config.LLM_MODEL
            )
            
            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            if "deepseek/deepseek-chat-v3-0324" in preferred_model:
                model_name = "DeepSeek Chat v3"
            elif "deepseek/deepseek-chat-v3" in preferred_model:
                model_name = "DeepSeek V3 0324"
            elif "deepseek/deepseek-r1" in preferred_model:
                model_name = "DeepSeek-R1"
            elif preferred_model.startswith("google/"):
                model_name = "Gemini 2.0 Flash"
            elif preferred_model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in preferred_model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-3.5-sonnet" in preferred_model:
                model_name = "Claude 3.5 Sonnet"
            elif "claude-3.7-sonnet" in preferred_model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in preferred_model:
                model_name = "Qwen QwQ 32B"
            elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in preferred_model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in preferred_model:
                model_name = "Anubis Pro 105B"
            elif "microsoft/phi-4-multimodal-instruct" in preferred_model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in preferred_model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in preferred_model:
                model_name = "Phi-3.5 Mini"
            
            # Create a description of all model strengths
            model_descriptions = [
                f"**Qwen QwQ 32B**: RECOMMENDED - Great for roleplaying with strong lore accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Prone to minor hallucinations due to it being a small model, and it sometimes slips in Chinese phrases. Uses ({bot.config.get_top_k_for_model('qwen/qwq-32b:free')}) search results.",
                f"**Gemini 2.0 Flash**: RECOMMENDED - Better for accurate citations, factual responses, document analysis, image viewing capabilities, and has very fast response times. Uses more search results ({bot.config.get_top_k_for_model('google/gemini-2.0-flash-001')}) for broader context.",
                f"**DeepSeek Chat V3 0324**: Great for roleplaying, creative responses, and in-character immersion, but may make things up due to its creativity. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324:free')}) search results with the free version, otherwise uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324')}).",
                f"**DeepSeek-R1**: Similar to V3, but with reasoning. Great for roleplaying, more creative responses, and in-character immersion, but sometimes may make things up due to its creativity (less than V3 due to its reasoning). With free version uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1:free')}) search results, otherwise uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1')}).",
                f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({bot.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}) for balanced context.",
                f"**Claude 3.5 Haiku**: Excellent for comprehensive lore analysis and nuanced understanding with creativity, and has image viewing capabilities. Also great for longer roleplays. Uses a moderate number of search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}) for balanced context.",
                f"**Claude 3.5 Sonnet**: Advanced model similar to Claude 3.7 Sonnet, may be more creative but less analytical (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-sonnet')}) to save money.",
                f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({bot.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                f"**Wayfarer 70B**: Optimized for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({bot.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({bot.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                #f"**Phi-4 Multimodal**: Microsoft's latest multimodal model with vision capabilities. It's not a good model. Uses ({bot.config.get_top_k_for_model('microsoft/phi-4-multimodal-instruct')}) search results.",
            ]
            
            response = f"*neural architecture scan complete!* Your currently selected model is **{model_name}**.\n\n**Model strengths:**\n"
            response += "\n".join(model_descriptions)
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
                
        except Exception as e:
            logger.error(f"Error getting preferred model: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while retrieving your preferred model.")

    @bot.tree.command(name="toggle_debug", description="Toggle debug mode to show model information in responses")
    async def toggle_debug(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            # Toggle debug mode and get the new state
            new_state = bot.user_preferences_manager.toggle_debug_mode(str(interaction.user.id))
            
            if new_state:
                await interaction.followup.send("*neural diagnostics activated!* Debug mode is now **ON**. Responses will show which model was used to generate them.")
            else:
                await interaction.followup.send("*neural diagnostics deactivated!* Debug mode is now **OFF**. Responses will no longer show model information.")
                
        except Exception as e:
            logger.error(f"Error toggling debug mode: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while toggling debug mode.")

    @bot.tree.command(name="help", description="Learn how to use Publicia and understand her capabilities and limitations")
    async def help_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            response = "# **PUBLICIA HELP GUIDE**\n\n"
            response += "*greetings, human! my genetically enhanced brain is ready to assist you with imperial knowledge. here's how to use my capabilities:*\n\n"
            
            # Core functionality
            response += "## **CORE FUNCTIONALITY**\n\n"
            response += "**🔍 Asking Questions**\n"
            response += "• **Mention me** in a message with your question about Ledus Banum 77 and Imperial lore\n"
            response += "• Use `/query` command for more structured questions (supports image URLs for analysis)\n"
            response += "• I'll search my knowledge base and provide answers with citations where possible\n"
            response += "• You can attach images directly to mentioned messages for visual analysis\n\n"
            response += "If you reply to a message and ping Publicia, she will be able to see the message you are replying to\n\n"
            
            # Knowledge Base
            response += "## **KNOWLEDGE BASE & LIMITATIONS**\n\n"
            response += "**📚 What I Know**\n"
            response += "• My knowledge is based on documents and images uploaded to my neural database\n"
            response += "• I specialize in Ledus Banum 77 (aka Tundra) lore and Imperial institutions\n"
            response += "• I can cite specific documents when providing information\n"
            response += "• I understand the Infinite Empire's structure, planes of existence, and Resonant Drilling\n\n"
            
            response += "**⚠️ What I Don't Know**\n"
            response += "• Information not contained in my document or image database\n"
            response += "• I cannot access the internet\n"
            response += "• I am bad at highly broad queries, or at queries asking for info that is not in my knowledge base\n"
            response += "• I am bad at queries that would require information from many different sources, as my embedding search system has a limit on the amount of document chunks it will return\n"
            response += "   • I would recommend breaking down your query into smaller, more focused questions so that my embeddings search can return more relevant and focused results\n"
            response += "• I may lack up to date information if my documents have not been updated\n\n"
            
            # How I Work
            response += "## **HOW I WORK**\n\n"
            response += "**🧠 Neural Processing**\n"
            response += "• I use semantic search with advanced reranking to find relevant information\n"
            response += "• I analyze your query to understand what you're looking for\n"
            response += "• I synthesize information from multiple documents when needed\n"
            response += "• I provide citations to document sources when possible\n"
            response += "• I automatically extract content from Google Docs linked in your queries\n"
            response += "• I use dynamic temperature control to adapt my responses to your query type\n\n"
            
            response += "**🖼️ Image Analysis**\n"
            response += "• I can analyze images in three ways:\n"
            response += "  - Attach an image directly when mentioning me\n"
            response += "  - Use `/query` with an image URL\n"
            response += "  - I can search my image database for relevant visual information\n"
            response += "• I can recognize content in images and integrate them into my responses\n"
            response += "• Add images to my knowledge base using `Publicia! add_image` for future searches\n"
            response += "• Vision-capable models: Gemini 2.0 Flash, Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet\n\n"
            
            # Document Management
            response += "## **DOCUMENT & IMAGE MANAGEMENT**\n\n"
            response += "**📚 Adding Information**\n"
            response += "• `/add_info` - Add text directly to my knowledge base\n"
            response += "• `Publicia! add_doc` - Add a document with an attachment\n"
            response += "• `/add_googledoc` - Connect a Google Doc to my knowledge base\n"
            response += "• `Publicia! add_image \"name\" [yes/no]` - Add an image with optional auto-description\n\n"
            
            response += "**📋 Managing Documents & Images**\n"
            response += "• `/list_docs` - See all documents in my knowledge base\n"
            response += "• `/list_images` - See all images in my visual knowledge base\n"
            response += "• `/list_files` - See all files in my knowledge base\n"
            response += "• `/retrieve_file` - Retrieve a file from my knowledge base and upload it to Discord\n\n"
            response += "• `/view_image` - View an image from my knowledge base\n"
            response += "• `/remove_doc` - Remove a document from my knowledge base\n"
            response += "• `/remove_image` - Remove an image from my knowledge base\n"
            response += "• `/remove_googledoc` - Disconnect a Google Doc\n"
            response += "• `/rename_document` - Rename a document in my database\n"
            response += "• `/search_docs` - Search directly in my document knowledge base\n"
            response += "• `/update_image_description` - Update the description for an image\n"
            response += "• `/reload_docs` - Reload all documents from disk (admin only)\n"
            response += "• `/archive_channel` - Archive messages from a Discord channel as a document (admin only)\n\n"
            
            # Conversation Management
            response += "## **CONVERSATION SYSTEM**\n\n"
            response += "**💬 How Conversations Work**\n"
            response += "• I remember your chats to provide more relevant, contextual responses\n"
            response += "• Each user has their own conversation history stored separately\n"
            response += "• When you ask something, I check your previous interactions for context\n"
            response += "• This lets me understand ongoing discussions, recurring topics, and your interests\n"
            response += "• Conversations are limited to the most recent 50 messages to maintain performance\n"
            response += "• Use `/parse_channel` to let me analyze recent channel messages for more context\n\n"

            response += "**🧠 Memory Management**\n"
            response += "• `/history [limit]` - See your recent conversation (default: shows last 10 messages)\n"
            response += "• `/manage_history [limit]` - View messages with numbered indices for selective deletion\n"
            response += "• `/delete_history_messages indices:\"0,2,5\" confirm:\"yes\"` - Remove specific messages by their indices\n"
            response += "• `/lobotomise` command to completely wipe your history\n"
            response += "• `/archive_conversation [archive_name]` - Save your current conversation history with optional custom name\n" 
            response += "• `/list_archives` - View all your archived conversations\n"
            response += "• `/swap_conversation archive_name` - Switch between current and archived conversations (automatically saves current conversation first)\n"
            response += "• `/delete_archive archive_name confirm:\"yes\"` - Permanently delete an archived conversation (requires confirmation)\n"
            response += "• Memory management lets you organize conversations, preserve important discussions, and start fresh when needed\n\n"
            
            # Customization
            response += "## **CUSTOMIZATION**\n\n"
            response += "**⚙️ AI Model Selection**\n"
            response += "• `/set_model` - Choose your preferred AI model:\n"
            response += "• `/get_model` - Check which model you're currently using, as well as get a list of all available models and their descriptions\n"
            response += "• `/toggle_debug` - Show/hide which model generated each response\n\n"
            response += "I recommend using Gemini 2.0 Flash for factual queries, DeepSeek-R1 for times when you want good prose and creative writing, Claude 3.5 Haiku for roleplay and accuracy, and QwQ 32B when you want a balance.\n\n"
            
            # Add our new section here
            response += "**🧪 Debugging Tools**\n"
            response += "• `/export_prompt` - Export the complete prompt for your query\n"
            response += "  - Shows system prompt, conversation history, search results, and more\n"
            response += "  - Helps understand exactly how I process your questions\n"
            response += "  - Includes privacy option to make output only visible to you\n\n"
            
            # Tips
            response += "## **TIPS FOR BEST RESULTS**\n\n"
            response += "• Ask specific questions for more accurate answers\n"
            response += "• If I don't know something, add relevant documents or images to my database\n"
            response += "• Use Google Docs integration for large, regularly updated documents\n"
            response += "• Include links to Google Docs in your queries for on-the-fly context\n"
            response += "• For creative writing, try DeepSeek-R1\n"
            response += "• For factual accuracy, try Gemini 2.0 Flash or Claude 3.5 Haiku\n\n"
            response += "*my genetically enhanced brain is always ready to help... just ask!*"
            
            # Send the response in chunks
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
                
        except Exception as e:
            logger.error(f"Error displaying help: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while trying to display help information.")

    @bot.tree.command(name="export_prompt", description="Export the full prompt that would be sent to the AI for your query")
    @app_commands.describe(
        question="The question to generate a prompt for",
        private="Whether to make the output visible only to you (default: True)"
    )
    async def export_prompt(interaction: discord.Interaction, question: str, private: bool = True):
        """Export the complete prompt that would be sent to the AI model."""
        await interaction.response.defer(ephemeral=private)
        try:
            # This command handles just like a regular query, but exports the prompt instead
            # of sending it to the AI model
            
            status_message = await interaction.followup.send(
                "*neural pathways activating... processing query for prompt export...*",
                ephemeral=private
            )
            
            # Get channel and user info
            channel_name = interaction.channel.name if interaction.guild else "DM"
            nickname = interaction.user.nick if (interaction.guild and interaction.user.nick) else interaction.user.name
            
            # Process exactly like a regular query until we have the messages
            conversation_messages = bot.conversation_manager.get_conversation_messages(interaction.user.name)
            
            await status_message.edit(content="*analyzing query...*")
            analysis = await bot.analyze_query(question)
            
            await status_message.edit(content="*searching imperial databases...*")
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id), 
                default_model=bot.config.LLM_MODEL
            )
            search_results = await bot.enhanced_search(question, analysis, preferred_model)
            
            await status_message.edit(content="*synthesizing information...*")
            synthesis = await bot.synthesize_results(question, search_results, analysis)
            
            # Now we have all the necessary information to create the prompt
            
            # Format the prompt for export
            await status_message.edit(content="*formatting prompt for export...*")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_content = f"""
    =========================================================
    PUBLICIA PROMPT EXPORT WITH CONTEXTUAL INFORMATION
    =========================================================
    Generated at: {timestamp}
    Query: {question}
    User: {nickname}
    Channel: {channel_name}
    Preferred model: {preferred_model}

    This file contains the complete prompt that would be sent to 
    the AI model when processing your query. This provides insight
    into how Publicia formulates responses by showing:

    1. The system prompt that defines Publicia's character
    2. Your conversation history for context
    3. Search results from relevant documents
    4. How your query was analyzed
    5. The final message containing your question

    SPECIAL FEATURE: This export shows whether document chunks have
    been enhanced with contextual information to improve search quality.

    NOTE: This export includes your conversation history, which
    may contain personal information. If you're sharing this file,
    please review the content first.
    =========================================================

    """
            
            # System prompt
            file_content += f"""
    SYSTEM PROMPT
    ---------------------------------------------------------
    This defines Publicia's character, abilities, and behavior.
    ---------------------------------------------------------
    {SYSTEM_PROMPT}
    =========================================================

    """
            
            # Conversation history
            if conversation_messages:
                file_content += f"""
    CONVERSATION HISTORY ({len(conversation_messages)} messages)
    ---------------------------------------------------------
    Previous messages provide context for your current query.
    ---------------------------------------------------------
    """
                for i, msg in enumerate(conversation_messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    file_content += f"[{i+1}] {role.upper()}: {content}\n\n"
                
                file_content += "=========================================================\n\n"
            
            # Synthesized context
            if synthesis:
                file_content += f"""
    SYNTHESIZED DOCUMENT CONTEXT
    ---------------------------------------------------------
    This is an AI-generated summary of the search results.
    ---------------------------------------------------------
    {synthesis}
    =========================================================

    """
            
            # Raw search results
            if search_results:
                file_content += f"""
    RAW SEARCH RESULTS ({len(search_results)} results)
    ---------------------------------------------------------
    These are the actual document chunks found by semantic search.
    Each result shows whether it has been enhanced with AI-generated context.
    ---------------------------------------------------------
    """
                for i, (doc, chunk, score, image_id, chunk_index, total_chunks) in enumerate(search_results):                        
                    # Check if this chunk has been contextualized
                    has_context = False
                    original_chunk = chunk
                    context_part = ""

                    # Check if we have contextualized chunks for this document
                    if hasattr(bot.document_manager, 'contextualized_chunks') and doc in bot.document_manager.contextualized_chunks:
                        # Check if we have enough contextualized chunks
                        if chunk_index - 1 < len(bot.document_manager.contextualized_chunks[doc]):
                            contextualized_chunk = bot.document_manager.contextualized_chunks[doc][chunk_index - 1]
                            
                            # Check if this is actually different from the original
                            if contextualized_chunk != original_chunk and original_chunk in contextualized_chunk:
                                has_context = True
                                # Try to extract just the context part (assuming it's prepended)
                                context_start_pos = contextualized_chunk.find(original_chunk)
                                if context_start_pos > 0:
                                    context_part = contextualized_chunk[:context_start_pos].strip()
                    
                    if image_id:
                        # Image result
                        image_name = bot.image_manager.metadata[image_id]['name'] if image_id in bot.image_manager.metadata else "Unknown Image"
                        file_content += f"[{i+1}] IMAGE: {image_name} (ID: {image_id}, score: {score:.2f})\n"
                        file_content += f"Description: {chunk}\n"
                    else:
                        # Document result
                        file_content += f"[{i+1}] DOCUMENT: {doc} (score: {score:.2f}, chunk {chunk_index}/{total_chunks})\n"
                        file_content += f"Content: {chunk}\n"
                    
                    # Display context information
                    if has_context:
                        file_content += f"[CONTEXTUAL ENHANCEMENT]: YES\n"
                        file_content += f"Added context: {context_part}\n\n"
                    else:
                        file_content += f"[CONTEXTUAL ENHANCEMENT]: NO\n\n"
                
                file_content += "=========================================================\n\n"
            
            # User query
            file_content += f"""
    USER QUERY
    ---------------------------------------------------------
    This is your actual question/message sent to Publicia.
    ---------------------------------------------------------
    {nickname}: {question}
    =========================================================

    """
            
            # Analysis data
            if analysis and analysis.get("success"):
                file_content += f"""
    QUERY ANALYSIS
    ---------------------------------------------------------
    This shows how your query was analyzed to improve search results.
    ---------------------------------------------------------
    {json.dumps(analysis, indent=2)}
    =========================================================
    """
            
            # Save to file
            file_name = f"publicia_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(file_content)
            
            # Check size and truncate if needed
            file_size = os.path.getsize(file_name)
            if file_size > 8 * 1024 * 1024:  # 8MB Discord limit
                truncated_file_name = f"publicia_prompt_truncated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(file_name, "r", encoding="utf-8") as f_in:
                    with open(truncated_file_name, "w", encoding="utf-8") as f_out:
                        f_out.write(f"WARNING: Original prompt was {file_size / (1024*1024):.2f}MB, exceeding Discord's limit. Content has been truncated.\n\n")
                        f_out.write(f_in.read(7 * 1024 * 1024))
                        f_out.write("\n\n[... Content truncated due to file size limits ...]")
                        
                os.remove(file_name)
                file_name = truncated_file_name
            
            # Upload file
            await status_message.edit(content="*uploading prompt file...*")
            await interaction.followup.send(
                file=discord.File(file_name), 
                content=f"*here's the full prompt that would be sent to the AI model for your query. this includes the system prompt, conversation history, and search results with contextual enhancement info:*",
                ephemeral=private
            )
            
            # Clean up
            os.remove(file_name)
            await status_message.edit(content="*prompt export complete!*")
            
        except Exception as e:
            logger.error(f"Error exporting prompt: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await interaction.followup.send("*neural circuit overload!* failed to export prompt due to an error.")

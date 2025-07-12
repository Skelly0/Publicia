"""
Utility commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import json
import os
from datetime import datetime, timedelta, timezone # Added timedelta, timezone
from typing import List, Dict, Any # Added typing imports
from utils.helpers import split_message
from prompts.system_prompt import SYSTEM_PROMPT, INFORMATIONAL_SYSTEM_PROMPT, get_system_prompt_with_documents # Added INFORMATIONAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all utility commands with the bot."""

    @bot.tree.command(name="list_commands", description="List all available commands")
    async def list_commands(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            response = "*Accessing command database through enhanced synapses...*\n\n"
            response += "# AVAILABLE COMMANDS\n\n"
            
            # SLASH COMMANDS SECTION
            response += "## Slash Commands (`/command`)\n\n"

            # Define standard command categories
            standard_categories = {
                "Lore Queries": ["query", "query_full_context"],
                "Document Management": ["list_docs", "search_docs", "search_keyword", "search_keyword_bm25", "list_googledocs", "list_googlesheets", "retrieve_file", "summarize_doc", "view_chunk"],
                "Image Management": ["list_images", "view_image"],
                "Utility": ["list_commands", "set_model", "get_model", "toggle_debug", "toggle_prompt_mode", "pronouns", "temperature", "help", "whats_new"],
                "Context/Memory Management": ["parse_channel", "history", "manage_history", "delete_history_messages", "swap_conversation", "list_archives", "archive_conversation", "delete_archive", "lobotomise", "memory_clear", "delete_history_messages"]
            }

            # Define admin-only command categories
            admin_categories = {
                "Document Management": [
                    "add_info",
                    "remove_doc",
                    "add_googledoc",
                    "remove_googledoc",
                    "rename_document",
                    "archive_channel",
                    "set_doc_channel",
                    "track_channel",
                    "untrack_channel",
                    "reload_docs",
                    "regenerate_embeddings",
                    "refresh_docs",
                    "add_googlesheet",
                    "remove_googlesheet",
                    "refresh_sheets",
                    "force_refresh_googlesheets",
                ],
                "Image Management": ["edit_image", "remove_image", "update_image_description"],
                "Utility": ["ban_user", "unban_user", "compare_models"],
            }

            # Define a set of admin command names for easy checking (not strictly needed for the new output structure, but fixing syntax)
            admin_command_names = {
                "add_info",
                "remove_doc",
                "add_googledoc",
                "remove_googledoc",
                "rename_document",
                "archive_channel",
                "set_doc_channel",
                "track_channel",
                "untrack_channel",
                "reload_docs",
                "regenerate_embeddings",
                "refresh_docs",
                "add_googlesheet",
                "remove_googlesheet",
                "refresh_sheets",
                "force_refresh_googlesheets",
                "edit_image",
                "remove_image",
                "update_image_description",
                "ban_user",
                "unban_user",
                "archive_conversation",
                "delete_archive",
                "lobotomise",
                "memory_clear",
                "delete_history_messages",
                "parse_channel",
                "compare_models"
            }

            # List standard commands first
            for category, cmd_list in standard_categories.items():
                response += f"__*{category}*__\n" # Standard category title

                # Sort the command list alphabetically
                sorted_cmd_list = sorted(cmd_list)

                for cmd_name in sorted_cmd_list:
                    cmd = bot.tree.get_command(cmd_name)
                    if cmd:
                        desc = cmd.description or "No description available"
                        response += f"`/{cmd_name}`: {desc}\n"
                response += "\n"

            # List admin-only commands in a separate section
            response += "## Admin Only Slash Commands\n\n"

            for category, cmd_list in admin_categories.items():
                response += f"__*{category}*__\n" # Admin subcategory title

                # Sort the command list alphabetically
                sorted_cmd_list = sorted(cmd_list)

                for cmd_name in sorted_cmd_list:
                    cmd = bot.tree.get_command(cmd_name)
                    if cmd:
                        desc = cmd.description or "No description available"
                        response += f"`/{cmd_name}`: {desc}\n" # No need for "(Admin Only)" marker here as it's in the section title
                response += "\n"

            # PREFIX COMMANDS SECTION
            response += "## Prefix Commands (`Publicia! command`)\n\n"
            
            # Get prefix commands from the bot
            prefix_commands = sorted(bot.commands, key=lambda x: x.name)
            
            # Group prefix commands by category (estimate categories based on names)
            # Assuming prefix commands are also admin-only for adding/removing info
            prefix_categories = {
                "Admin Only (Document Management)": [],
                "Admin Only (Image Management)": []
            }
            
            # Sort commands into categories
            for cmd in prefix_commands:
                if "doc" in cmd.name.lower():
                    prefix_categories["Admin Only (Document Management)"].append(cmd)
                elif "image" in cmd.name.lower():
                    prefix_categories["Admin Only (Image Management)"].append(cmd)
            
            # Format and add each category of prefix commands
            for category, cmds in prefix_categories.items():
                if cmds:  # Only show categories that have commands
                    response += f"__***{category}***__\n"
                    for cmd in cmds:
                        brief = cmd.brief or "No description available"
                        response += f"`Publicia! {cmd.name}`: {brief}\n"
                    response += "\n"
            
            response += "\n*You can ask questions about ledus banum 77 and imperial lore by pinging/mentioning me or using the /query command!*"
            response += "\n*My genetically enhanced brain is always ready to help... just ask!*"
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
        except Exception as e:
            logger.error("Error listing commands: %s", e)
            await interaction.followup.send("*my enhanced neurons misfired!* couldn't retrieve command list right now...")

    @bot.tree.command(name="set_model", description="Set your preferred AI model for responses")
    @app_commands.describe(model="Choose the AI model you prefer")
    @app_commands.choices(model=[
        app_commands.Choice(name="MiniMax M1", value="minimax/minimax-m1"),
        app_commands.Choice(name="OpenAI o4 Mini", value="openai/o4-mini"),
        app_commands.Choice(name="Gemini 2.5 Flash", value="google/gemini-2.5-flash-preview:thinking"),
        #app_commands.Choice(name="Gemini 2.5 Pro", value="google/gemini-2.5-pro-preview-03-25"), # Added new model
        app_commands.Choice(name="Qwen QwQ 32B", value="qwen/qwq-32b"),
        app_commands.Choice(name="Qwen 3 235B A22B", value="qwen/qwen3-235b-a22b"),
        app_commands.Choice(name="DeepSeek V3 0324", value="deepseek/deepseek-chat-v3-0324:floor"), # Added as per request
        app_commands.Choice(name="DeepSeek-R1", value="deepseek/deepseek-r1-0528"),
        app_commands.Choice(name="Claude 3.5 Haiku", value="anthropic/claude-3.5-haiku"),
        app_commands.Choice(name="Claude 4 Sonnet", value="anthropic/claude-sonnet-4"),
        app_commands.Choice(name="Nous: Hermes 405B", value="nousresearch/hermes-3-llama-3.1-405b"),
        app_commands.Choice(name="Kimi K2", value="moonshotai/kimi-k2"),
        #app_commands.Choice(name="Claude 3.7 Sonnet", value="anthropic/claude-3.7-sonnet"),
        #app_commands.Choice(name="Testing Model", value="eva-unit-01/eva-qwen-2.5-72b"),
        #app_commands.Choice(name="Wayfarer 70B", value="latitudegames/wayfarer-large-70b-llama-3.3"),
        #app_commands.Choice(name="Anubis Pro 105B", value="thedrummer/anubis-pro-105b-v1"),
        #app_commands.Choice(name="Llama 4 Maverick", value="meta-llama/llama-4-maverick:floor"),
        #app_commands.Choice(name="Grok 3 Mini", value="x-ai/grok-3-mini-beta"),
        #app_commands.Choice(name="OpenAI GPT-4.1 Mini", value="openai/gpt-4.1-mini"),
        #app_commands.Choice(name="OpenAI GPT-4.1 Nano", value="openai/gpt-4.1-nano"),
        #app_commands.Choice(name="Phi-4 Multimodal", value="microsoft/phi-4-multimodal-instruct"),
    ])
    async def set_model(interaction: discord.Interaction, model: str):
        await interaction.response.defer()
        try:
            # Check if user is allowed to use Claude 3.7 Sonnet
            if model == "anthropic/claude-3.7-sonnet" and str(interaction.user.id) != "203229662967627777":
                await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                return

            if model == "anthropic/claude-sonnet-4" and str(interaction.user.id) != "203229662967627777":
                await interaction.followup.send("*neural access denied!* Claude 3.7 Sonnet is restricted to administrators only.")
                return
            
            if model == "google/gemini-2.5-pro-preview-03-25" and str(interaction.user.id) != "203229662967627777":
                await interaction.followup.send("*neural access denied!* Gemini 2.5 Pro is restricted to administrators only.")
                return
                
            success = bot.user_preferences_manager.set_preferred_model(str(interaction.user.id), model)
            
            # Get friendly model name based on the model value
            model_name = "Unknown Model"
            # Consolidated DeepSeek V3 check
            if "deepseek/deepseek-chat-v3" in model:
                model_name = "DeepSeek V3 0324"
            elif "deepseek/deepseek-r1" in model:
                model_name = "DeepSeek-R1"
            elif "meta-llama/llama-4-maverick" in model:
                model_name = "Llama 4 Maverick"
            elif "google/gemini-2.5-pro-preview-03-25" in model:
                model_name = "Gemini 2.5 Pro"
            elif "google/gemini-2.5-flash" in model:
                model_name = "Gemini 2.5 Flash"
            elif model.startswith("google/"): # Keep this as a fallback for other google models
                model_name = "Gemini 2.0 Flash"
            elif model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-sonnet-4" in model:
                model_name = "Claude 4 Sonnet"
            elif "claude-3.7-sonnet" in model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in model:
                model_name = "Qwen QwQ 32B"
            elif "qwen/qwen3-235b-a22b" in model:
                model_name = "Qwen 3 235B A22B"
            elif "moonshotai/kimi-k2" in model:
                model_name = "Kimi K2"
            elif "unslopnemo" in model or "eva-unit-01/eva-qwen-2.5-72b" in model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in model:
                model_name = "Anubis Pro 105B"
            elif "x-ai/grok-3-mini-beta" in model:
                model_name = "Grok 3 Mini"
            elif "microsoft/phi-4-multimodal-instruct" in model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in model:
                model_name = "Phi-3.5 Mini"
            elif model == "openai/gpt-4.1-mini":
                model_name = "OpenAI GPT-4.1 Mini"
            elif model == "openai/gpt-4.1-nano":
                model_name = "OpenAI GPT-4.1 Nano"
            elif model == "openai/o4-mini":
                model_name = "OpenAI o4 Mini"
            elif model == "minimax/minimax-m1":
                model_name = "MiniMax M1"
            
            if success:
                # Create a description of all model strengths
                # Create a description of all model strengths
                model_descriptions = [
                    f"**MiniMax M1**: __RECOMMENDED__ - A large-scale, open-weight reasoning model from MiniMax, good for general tasks and long-context understanding. Great for finding accurate information. Good prompt adherence and an interesting personality. Uses ({bot.config.get_top_k_for_model('minimax/minimax-m1')}) search results.",
                    f"**OpenAI o4 Mini**: __RECOMMENDED__ An OpenAI model that is very good for factual accuracy, avoiding hallucinations, and speed of response. Uses ({bot.config.get_top_k_for_model('openai/o4-mini')}) search results.",
                    f"**Gemini 2.5 Flash**: - Fine for prompt adherence, accurate citations, image viewing capabilities, and fast response times. Prone to hallucinating if asked about something not in it's supplied documents. Uses more search results ({bot.config.get_top_k_for_model('google/gemini-2.5-flash-preview:thinking')}).",
                    #f"**Gemini 2.5 Pro**: (admin only) Uses ({bot.config.get_top_k_for_model('google/gemini-2.5-pro-preview-03-25')}) search results.",
                    f"**Qwen QwQ 32B**: __RECOMMENDED__ - Great for roleplaying and creativity with strong factual accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Uses ({bot.config.get_top_k_for_model('qwen/qwq-32b')}) search results.",
                    f"**Qwen 3 235B A22B**: Uses ({bot.config.get_top_k_for_model('qwen/qwen3-235b-a22b')}) search results.",
                    #f"**Gemini 2.5 Pro Exp**: Experimental Pro model, potentially stronger reasoning and generation than Flash, includes vision. Uses ({bot.config.get_top_k_for_model('google/gemini-2.5-pro-exp-03-25')}) search results.", # Added new model description
                    f"**DeepSeek V3 0324**: Great for roleplaying, creative responses, and in-character immersion, but often makes things up due to its creativity. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324')}) search results.", # Added as per request
                    f"**DeepSeek-R1**: Uses the 0528 version. Similar to V3 0324 but with reasoning. Great at creative writing, roleplaying, more creative responses, and in-character immersion, but sometimes may make things up due to its creativity. Often factually inaccurate. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1-0528')}) search results.", # Updated description slightly for clarity
                    f"**Claude 3.5 Haiku**: __RECOMMENDED__ - A good balance between creativity and accuracy, and has image viewing capabilities. One of the best for longer roleplays. Uses a moderate number of search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}).",
                    f"**Claude 4 Sonnet**: Advanced model similar to Claude 3.5 Haiku, the best model (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-sonnet-4')}).",
                    #f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                    f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({bot.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}).",
                    f"**Kimi K2**: Large-scale Mixture-of-Experts model from Moonshot AI with 1 trillion parameters (32B active per forward pass), great for creative writing (not tested much yet). Uses ({bot.config.get_top_k_for_model('moonshotai/kimi-k2')}) search results.",
                    #f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({bot.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                    #f"**Wayfarer 70B**: A model finetuned for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({bot.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                    #f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Supposedly excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({bot.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                    #f"**Llama 4 Maverick**: Good for prompt adherence and factual responses. Pretty good at roleplaying, if a bit boring. Uses ({bot.config.get_top_k_for_model('meta-llama/llama-4-maverick:floor')}) search results.",
                    #f"**Grok 3 Mini**: __RECOMMENDED__ - An intelligent small model, good for factual responses, prompt adherence, character acting, and interesting speaking style. Uses ({bot.config.get_top_k_for_model('x-ai/grok-3-mini-beta')}) search results.",
                    #f"**OpenAI GPT-4.1 Mini**: A compact and efficient model from OpenAI, good for general tasks. Uses ({bot.config.get_top_k_for_model('openai/gpt-4.1-mini')}) search results.",
                    #f"**OpenAI GPT-4.1 Nano**: An even smaller OpenAI model, optimized for speed and efficiency. Uses ({bot.config.get_top_k_for_model('openai/gpt-4.1-nano')}) search results.",
                ]
                
                response = f"*neural architecture reconfigured!* Your preferred model has been set to **{model_name}**.\n\n**Model strengths:**\n"
                response += "\n".join(model_descriptions)
                
                for chunk in split_message(response):
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send("*synaptic error detected!* Failed to set your preferred model. Please try again later.")
                
        except Exception as e:
            logger.error("Error setting preferred model: %s", e)
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
            # Consolidated DeepSeek V3 check
            if "deepseek/deepseek-chat-v3" in preferred_model:
                model_name = "DeepSeek V3 0324"
            elif "deepseek/deepseek-r1" in preferred_model:
                model_name = "DeepSeek-R1"
            elif "meta-llama/llama-4-maverick" in preferred_model:
                model_name = "Llama 4 Maverick"
            elif "google/gemini-2.5-pro-preview-03-25" in preferred_model:
                model_name = "Gemini 2.5 Pro"
            elif "google/gemini-2.5-flash" in preferred_model:
                model_name = "Gemini 2.5 Flash"
            elif preferred_model.startswith("google/"): # Keep this as a fallback for other google models
                model_name = "Gemini 2.0 Flash"
            elif preferred_model.startswith("nousresearch/"):
                model_name = "Nous: Hermes 405B Instruct"
            elif "claude-3.5-haiku" in preferred_model:
                model_name = "Claude 3.5 Haiku"
            elif "claude-sonnet-4" in preferred_model:
                model_name = "Claude 4 Sonnet"
            elif "claude-3.7-sonnet" in preferred_model:
                model_name = "Claude 3.7 Sonnet"
            elif "qwen/qwq-32b" in preferred_model:
                model_name = "Qwen QwQ 32B"
            elif "qwen/qwen3-235b-a22b" in preferred_model:
                model_name = "Qwen 3 235B A22B"
            elif "moonshotai/kimi-k2" in preferred_model:
                model_name = "Kimi K2"
            elif "unslopnemo" in preferred_model or "eva-unit-01/eva-qwen-2.5-72b" in preferred_model:
                model_name = "Testing Model"
            elif "latitudegames/wayfarer" in preferred_model:
                model_name = "Wayfarer 70B"
            elif "thedrummer/anubis-pro" in preferred_model:
                model_name = "Anubis Pro 105B"
            elif "x-ai/grok-3-mini-beta" in preferred_model:
                model_name = "Grok 3 Mini"
            elif "microsoft/phi-4-multimodal-instruct" in preferred_model:
                model_name = "Phi-4 Multimodal"
            elif "microsoft/phi-4" in preferred_model:
                model_name = "Phi-4"
            elif "microsoft/phi-3.5-mini-128k-instruct" in preferred_model:
                model_name = "Phi-3.5 Mini"
            elif preferred_model == "openai/gpt-4.1-mini":
                model_name = "OpenAI GPT-4.1 Mini"
            elif preferred_model == "openai/gpt-4.1-nano":
                model_name = "OpenAI GPT-4.1 Nano"
            elif preferred_model == "openai/o4-mini":
                model_name = "OpenAI o4 Mini"
            elif preferred_model == "minimax/minimax-m1":
                model_name = "MiniMax M1"
            
            # Create a description of all model strengths
            model_descriptions = [
                    f"**MiniMax M1**: __RECOMMENDED__ - A large-scale, open-weight reasoning model from MiniMax, good for general tasks and long-context understanding. Great for finding accurate information. Good prompt adherence and an interesting personality. Uses ({bot.config.get_top_k_for_model('minimax/minimax-m1')}) search results.",
                    f"**OpenAI o4 Mini**: __RECOMMENDED__ An OpenAI model that is very good for factual accuracy, avoiding hallucinations, and speed of response. Uses ({bot.config.get_top_k_for_model('openai/o4-mini')}) search results.",
                    f"**Gemini 2.5 Flash**: - Fine for prompt adherence, accurate citations, image viewing capabilities, and fast response times. Prone to hallucinating if asked about something not in it's supplied documents. Uses more search results ({bot.config.get_top_k_for_model('google/gemini-2.5-flash-preview:thinking')}).",
                    #f"**Gemini 2.5 Pro**: (admin only) Uses ({bot.config.get_top_k_for_model('google/gemini-2.5-pro-preview-03-25')}) search results.",
                    f"**Qwen QwQ 32B**: __RECOMMENDED__ - Great for roleplaying and creativity with strong factual accuracy and in-character immersion. Produces detailed, nuanced responses with structured formatting. Uses ({bot.config.get_top_k_for_model('qwen/qwq-32b')}) search results.",
                    f"**Qwen 3 235B A22B**: Uses ({bot.config.get_top_k_for_model('qwen/qwen3-235b-a22b')}) search results.",
                    #f"**Gemini 2.5 Pro Exp**: Experimental Pro model, potentially stronger reasoning and generation than Flash, includes vision. Uses ({bot.config.get_top_k_for_model('google/gemini-2.5-pro-exp-03-25')}) search results.", # Added new model description
                    f"**DeepSeek V3 0324**: Great for roleplaying, creative responses, and in-character immersion, but often makes things up due to its creativity. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-chat-v3-0324')}) search results.", # Added as per request
                    f"**DeepSeek-R1**: Uses the 0528 version. Similar to V3 0324 but with reasoning. Great at creative writing, roleplaying, more creative responses, and in-character immersion, but sometimes may make things up due to its creativity. Often factually inaccurate. Uses ({bot.config.get_top_k_for_model('deepseek/deepseek-r1-0528')}) search results.", # Updated description slightly for clarity
                    f"**Claude 3.5 Haiku**: __RECOMMENDED__ - A good balance between creativity and accuracy, and has image viewing capabilities. One of the best for longer roleplays. Uses a moderate number of search results ({bot.config.get_top_k_for_model('anthropic/claude-3.5-haiku')}).",
                    f"**Claude 4 Sonnet**: Advanced model similar to Claude 3.5 Haiku, the best model (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-sonnet-4')}).",
                    #f"**Claude 3.7 Sonnet**: Most advanced model, combines creative and analytical strengths (admin only). Uses fewer search results ({bot.config.get_top_k_for_model('anthropic/claude-3.7-sonnet')}) to save money.",
                    f"**Nous: Hermes 405B**: Great for roleplaying. Balanced between creativity and accuracy. Uses a moderate number of search results ({bot.config.get_top_k_for_model('nousresearch/hermes-3-llama-3.1-405b')}).",
                    f"**Kimi K2**: Large-scale Mixture-of-Experts model from Moonshot AI with 1 trillion parameters (32B active per forward pass), great for creative writing (not tested much yet). Uses ({bot.config.get_top_k_for_model('moonshotai/kimi-k2')}) search results.",
                    #f"**Testing Model**: Currently using EVA Qwen2.5 72B, a narrative-focused model. Uses ({bot.config.get_top_k_for_model('eva-unit-01/eva-qwen-2.5-72b')}) search results. This model can be easily swapped to test different OpenRouter models.",
                    #f"**Wayfarer 70B**: A model finetuned for narrative-driven roleplay with realistic stakes and conflicts. Good for immersive storytelling and character portrayal. Uses ({bot.config.get_top_k_for_model('latitudegames/wayfarer-large-70b-llama-3.3')}) search results.",
                    #f"**Anubis Pro 105B**: 105B parameter model with enhanced emotional intelligence and creativity. Supposedly excels at nuanced character portrayal and superior prompt adherence as compared to smaller models. Uses ({bot.config.get_top_k_for_model('thedrummer/anubis-pro-105b-v1')}) search results.",
                    #f"**Llama 4 Maverick**: Good for prompt adherence and factual responses. Pretty good at roleplaying, if a bit boring. Uses ({bot.config.get_top_k_for_model('meta-llama/llama-4-maverick:floor')}) search results.",
                    #f"**Grok 3 Mini**: __RECOMMENDED__ - An intelligent small model, good for factual responses, prompt adherence, character acting, and interesting speaking style. Uses ({bot.config.get_top_k_for_model('x-ai/grok-3-mini-beta')}) search results.",
                    #f"**OpenAI GPT-4.1 Mini**: A compact and efficient model from OpenAI, good for general tasks. Uses ({bot.config.get_top_k_for_model('openai/gpt-4.1-mini')}) search results.",
                    #f"**OpenAI GPT-4.1 Nano**: An even smaller OpenAI model, optimized for speed and efficiency. Uses ({bot.config.get_top_k_for_model('openai/gpt-4.1-nano')}) search results.",
            ]
            
            response = f"*neural architecture scan complete!* Your currently selected model is **{model_name}**.\n\n**Model strengths:**\n"
            response += "\n".join(model_descriptions)
            
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
                
        except Exception as e:
            logger.error("Error getting preferred model: %s", e)
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
            logger.error("Error toggling debug mode: %s", e)
            await interaction.followup.send("*neural circuit overload!* An error occurred while toggling debug mode.")

    @bot.tree.command(name="toggle_prompt_mode", description="Toggle between standard (immersive) and informational (concise) system prompts")
    async def toggle_prompt_mode(interaction: discord.Interaction):
        """Toggles the system prompt mode for the user."""
        await interaction.response.defer()
        try:
            # Toggle the mode using the preference manager
            new_state = bot.user_preferences_manager.toggle_informational_prompt_mode(str(interaction.user.id))

            if new_state:
                await interaction.followup.send("*neural pathways adjusted!* Informational prompt mode is now **ON**. Responses will be concise and factual, without roleplaying.")
            else: # Informational mode is OFF (Standard mode is ON)
                await interaction.followup.send("*neural pathways restored!* Standard prompt mode is now **ON** (Informational mode is **OFF**). Responses will be immersive and in-character.")

        except Exception as e:
            logger.error("Error toggling informational prompt mode: %s", e)
            await interaction.followup.send("*neural circuit overload!* An error occurred while toggling the prompt mode.")

    @bot.tree.command(name="help", description="Learn how to use Publicia and understand her capabilities and limitations")
    async def help_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            response = "# **PUBLICIA HELP GUIDE**\n\n"
            response += "Greetings! I am a Discord bot that roleplays as the abhuman Publicia. I act as a lore repository for our world and setting that can be queried.\n"
            response += "My purpose is primarily to answer questions about the lore of Ledus Banum 77 and the Infinite Empire (the setting of Season 7/this Discord roleplay server). I am helpful for trying to find information about the various natives of the planet or about the Empire. I can also engage in roleplay!\n"
            response += "DISCLAIMER: I can sometimes hallucinate false information or get confabulated, so for anything important, double check the original documents or ask someone to help!\n"
            response += "\nHere's a quick overview of my functionality, limitations, and commands:\n\n"

            # Core functionality
            response += "## **CORE FUNCTIONALITY**\n\n"
            response += "**🔍 Asking Questions:**\n"
            response += "- **Mention me** (@Publicia) with your question about Ledus Banum 77 / Imperial lore.\n"
            response += "- Use `/query` for structured questions (supports image URLs).\n"
            response += "- Use `/query_full_context` for deep dives using all documents (limited use).\n"
            response += "- Attach images directly when mentioning me for visual analysis.\n"
            response += "- Including a Google Doc link when mentiong me me will provide that document as context to me.\n"
            response += "- Use `/parse_channel` to make it so that I can see recent Discord messages in the channel you are mentioning me in.\n"
            response += "- Replying to a message while mentioning me provides that replied message and any image attachments with it as context to me.\n\n"

            # How I Work & Limitations
            response += "## **HOW I WORK & LIMITATIONS**\n\n"
            response += "**🧠 Neural Processing:** I use semantic search on my internal knowledge base (documents & images) to find relevant info. I can synthesize answers and cite sources. When mentioning me I remember our conversation history for context (last 50 messages).\n\n"
            response += "**🖼️ Image Analysis:** I can analyze images attached to mentions, via `/query` URLs, or from my internal image database (if the model supports vision, e.g., Gemini Flash, Claude Haiku).\n\n"
            response += "**⚠️ Limitations:** My knowledge is limited to uploaded documents/images. I cannot access the internet. Very broad queries or questions requiring info from many sources might yield poor results; try breaking down complex questions.\n\n"

            # Key Command Areas
            response += "## **KEY COMMAND AREAS**\n\n"
            response += "I offer commands for:\n"
            response += "- **Document & Image Management:** Adding, listing, removing, searching, summarizing docs/images.\n"
            response += "- **Conversation Management:** Viewing, deleting, archiving, and swapping conversation history.\n"
            response += "- **Customization:** Setting preferred AI models, toggling debug/prompt modes, setting your preferred pronouns, and customizing temperature ranges.\n\n"
            response += "**For a full list of all commands and their descriptions, please use the `/list_commands` command.**\n\n"

            # Tips
            response += "## **TIPS FOR BEST RESULTS**\n\n"
            response += "- Avoid queries that are too specific, or too vague/broad. A vague query will not let me access enough relevant documents (try using `/query_full_context` for those sorts of questions).\n"
            response += "- I use a semantic search for finding information from my documents, and so a best practice is to include keywords in your queries that help point me to the correct documents.\n"
            response += "- If you want me to speak more concisely and drop the roleplay, use the command `/toggle_prompt_mode`.\n"
            response += "- Ask my creator skellia if you require help or explanations for anything.\n"
            response += "- Ask admins to add additional info (character backstories and so on) that you have written to expand my knowledge.\n"
            response += "- Choose models based on your needs (e.g., Qwen QwQ for a mix of accuracy and creative flair, Claude Haiku for longform RP).\n\n"

            response += "*My genetically enhanced brain is always ready to help... just ask!*"

            # Send the response in chunks
            for chunk in split_message(response):
                await interaction.followup.send(chunk)

        except Exception as e:
            logger.error("Error displaying help: %s", e)
            await interaction.followup.send("*neural circuit overload!* An error occurred while trying to display help information.")

    """@bot.tree.command(name="export_prompt", description="Export the full prompt that would be sent to the AI for your query")
    @app_commands.describe(
        question="The question to generate a prompt for",
        private="Whether to make the output visible only to you (default: True)"
    )
    async def export_prompt(interaction: discord.Interaction, question: str, private: bool = True):
        #Export the complete prompt that would be sent to the AI model.
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
            channel_description = getattr(interaction.channel, "topic", None)
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
            file_content = f#
    =========================================================
    PUBLICIA PROMPT EXPORT WITH CONTEXTUAL INFORMATION
    =========================================================
    Generated at: {timestamp}
    Query: {question}
    User: {nickname}
    Channel: {channel_name}
    {f'The channel has the description: {channel_description}\n' if channel_description else ''}
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

    #
            
            # System prompt
            file_content += f#
    SYSTEM PROMPT
    ---------------------------------------------------------
    This defines Publicia's character, abilities, and behavior.
    ---------------------------------------------------------
    {get_system_prompt_with_documents(bot.document_manager.get_document_list_content())}
    =========================================================

    #
            
            # Conversation history
            if conversation_messages:
                file_content += f#
    CONVERSATION HISTORY ({len(conversation_messages)} messages)
    ---------------------------------------------------------
    Previous messages provide context for your current query.
    ---------------------------------------------------------
    #
                for i, msg in enumerate(conversation_messages):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    file_content += f"[{i+1}] {role.upper()}: {content}\n\n"
                
                file_content += "=========================================================\n\n"
            
            # Synthesized context
            if synthesis:
                file_content += f#
    SYNTHESIZED DOCUMENT CONTEXT
    ---------------------------------------------------------
    This is an AI-generated summary of the search results.
    ---------------------------------------------------------
    {synthesis}
    =========================================================

    #
            
            # Raw search results
            if search_results:
                file_content += f#
    RAW SEARCH RESULTS ({len(search_results)} results)
    ---------------------------------------------------------
    These are the actual document chunks found by semantic search.
    Each result shows whether it has been enhanced with AI-generated context.
    ---------------------------------------------------------
    #
                for i, (doc, chunk, score, image_id, chunk_index, total_chunks) in enumerate(search_results):                        
                    # Check if this chunk has been contextualized
                    has_context = False
                    original_chunk = chunk
                    context_part = ""

                    # Check if we have contextualized chunks for this document and if they're enabled
                    use_contextualised = bot.config.USE_CONTEXTUALISED_CHUNKS if hasattr(bot, 'config') and hasattr(bot.config, 'USE_CONTEXTUALISED_CHUNKS') else True
                    
                    if use_contextualised and hasattr(bot.document_manager, 'contextualized_chunks') and doc in bot.document_manager.contextualized_chunks:
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
            file_content += f#
    USER QUERY
    ---------------------------------------------------------
    This is your actual question/message sent to Publicia.
    ---------------------------------------------------------
    {nickname}: {question}
    =========================================================

    #
            
            # Analysis data
            if analysis and analysis.get("success"):
                file_content += f#
    QUERY ANALYSIS
    ---------------------------------------------------------
    This shows how your query was analyzed to improve search results.
    ---------------------------------------------------------
    {json.dumps(analysis, indent=2)}
    =========================================================
    #
            
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
            logger.error("Error exporting prompt: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            await interaction.followup.send("*neural circuit overload!* failed to export prompt due to an error.")
    """

    @bot.tree.command(name="whats_new", description="Shows documents and images added or updated recently")
    @app_commands.describe(days="How many days back to check (default: 7)")
    async def whats_new(interaction: discord.Interaction, days: int = 7):
        """Summarizes recently added/updated documents and images."""
        await interaction.response.defer() # Defer response

        if days <= 0:
            await interaction.followup.send("Please provide a positive number of days.")
            return

        # Access managers via the bot instance passed to register_commands
        if not hasattr(bot, 'document_manager') or not hasattr(bot, 'image_manager'):
             await interaction.followup.send("*critical system error!* Document or Image Manager not available.")
             logger.error("whats_new command failed: Manager instances not found on bot.")
             return

        now = datetime.now(timezone.utc) # Use timezone-aware datetime
        cutoff_time = now - timedelta(days=days)
        recent_items = [] # Stores {'type': 'doc'/'image', 'name': ..., 'timestamp': ..., 'action': ..., 'id': ... (for images)}

        doc_manager = bot.document_manager
        img_manager = bot.image_manager

        # --- Process Documents ---
        try:
            # Assuming managers load metadata on init or have a method to ensure it's loaded
            if not doc_manager.metadata:
                 logger.warning("Document metadata is empty during whats_new check.")

            for doc_uuid, meta in doc_manager.metadata.items():
                # Get original name from UUID
                doc_name = doc_manager._get_original_name(doc_uuid)

                # Skip internal list document
                if hasattr(doc_manager, '_internal_list_doc_name') and doc_name == doc_manager._internal_list_doc_name:
                    continue

                added_ts_str = meta.get('added')
                updated_ts_str = meta.get('updated') # Check for 'updated' timestamp

                # Parse timestamps safely
                added_dt = None
                if added_ts_str:
                    try:
                        # Handle potential 'Z' suffix if present
                        if added_ts_str.endswith('Z'):
                            added_ts_str = added_ts_str[:-1] + '+00:00'
                        added_dt = datetime.fromisoformat(added_ts_str)
                        # Ensure timezone aware (assume UTC if naive)
                        if added_dt.tzinfo is None:
                            added_dt = added_dt.replace(tzinfo=timezone.utc)
                        else:
                            added_dt = added_dt.astimezone(timezone.utc)
                    except ValueError:
                        logger.warning("Could not parse 'added' timestamp '%s' for doc '%s'", added_ts_str, doc_name)

                updated_dt = None
                if updated_ts_str:
                     try:
                         # Handle potential 'Z' suffix if present
                         if updated_ts_str.endswith('Z'):
                             updated_ts_str = updated_ts_str[:-1] + '+00:00'
                         updated_dt = datetime.fromisoformat(updated_ts_str)
                         # Ensure timezone aware (assume UTC if naive)
                         if updated_dt.tzinfo is None:
                             updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                         else:
                             updated_dt = updated_dt.astimezone(timezone.utc)
                     except ValueError:
                         logger.warning("Could not parse 'updated' timestamp '%s' for doc '%s'", updated_ts_str, doc_name)

                # Determine the most recent relevant timestamp and action
                latest_dt = None
                action = "unknown"
                is_recent = False

                # Prioritize 'updated' if it's recent
                if updated_dt and updated_dt >= cutoff_time:
                    latest_dt = updated_dt
                    action = "updated"
                    is_recent = True
                # Otherwise, check 'added' if it's recent
                elif added_dt and added_dt >= cutoff_time:
                    latest_dt = added_dt
                    action = "added"
                    is_recent = True

                if is_recent and latest_dt:
                    is_image_doc = doc_name.startswith("image_") and doc_name.endswith(".txt")
                    if not is_image_doc:
                        # Regular document
                        recent_items.append({
                            'type': 'doc',
                            'name': doc_name, # This is now the original name
                            'timestamp': latest_dt,
                            'action': action,
                            'id': doc_uuid # Pass the UUID for the tracker
                        })
                    else:
                        # Image description document update/add
                        image_id = meta.get('image_id')
                        image_name = meta.get('image_name', 'Unknown Image')
                        if image_id:
                            # Check if this image is already in the list from its own 'added' timestamp
                            existing_item = next((item for item in recent_items if item.get('type') == 'image' and item.get('id') == image_id), None)
                            if existing_item:
                                # If description update is more recent, update the existing entry
                                if latest_dt > existing_item['timestamp']:
                                    existing_item['timestamp'] = latest_dt
                                    existing_item['action'] = f'description {action}' # e.g., "description updated"
                            else:
                                # Add new entry for the image based on description update
                                recent_items.append({
                                    'type': 'image',
                                    'id': image_id,
                                    'name': image_name,
                                    'timestamp': latest_dt,
                                    'action': f'description {action}'
                                })

        except Exception as e:
            logger.error("Error processing document metadata for whats_new: %s", e, exc_info=True)
            # Send error but continue to images
            await interaction.followup.send(f"*neural pathway disruption!* An error occurred while checking documents: {e}", ephemeral=True)


        # --- Process Images (for 'added' timestamp) ---
        try:
            if not img_manager.metadata:
                 logger.warning("Image metadata is empty during whats_new check.")

            for image_id, meta in img_manager.metadata.items():
                added_ts_str = meta.get('added')
                added_dt = None
                if added_ts_str:
                    try:
                        # Handle potential 'Z' suffix if present
                        if added_ts_str.endswith('Z'):
                            added_ts_str = added_ts_str[:-1] + '+00:00'
                        added_dt = datetime.fromisoformat(added_ts_str)
                        # Ensure timezone aware (assume UTC if naive)
                        if added_dt.tzinfo is None:
                            added_dt = added_dt.replace(tzinfo=timezone.utc)
                        else:
                            added_dt = added_dt.astimezone(timezone.utc)
                    except ValueError:
                        logger.warning("Could not parse 'added' timestamp '%s' for image '%s'", added_ts_str, image_id)

                if added_dt and added_dt >= cutoff_time:
                    # Check if already added via document update
                    existing_item = next((item for item in recent_items if item.get('type') == 'image' and item.get('id') == image_id), None)

                    if existing_item:
                        # If image was added *more* recently than description update, update timestamp/action
                        if added_dt > existing_item['timestamp']:
                            existing_item['timestamp'] = added_dt
                            existing_item['action'] = 'added' # Overwrite "description updated" if direct add is newer
                    else:
                        # Image added recently, wasn't found via description update
                        recent_items.append({
                            'type': 'image',
                            'id': image_id,
                            'name': meta.get('name', 'Unknown Image'),
                            'timestamp': added_dt,
                            'action': 'added'
                        })
        except Exception as e:
            logger.error("Error processing image metadata for whats_new: %s", e, exc_info=True)
            # Send error but continue to formatting
            await interaction.followup.send(f"*visual cortex error!* An error occurred while checking images: {e}", ephemeral=True)


        # --- Sort and Format Output ---
        if not recent_items:
            await interaction.followup.send(f"*database scan complete.* No documents or images were added or updated in the last {days} days.")
            return

        # Sort by timestamp descending
        recent_items.sort(key=lambda x: x['timestamp'], reverse=True)

        output_lines = [f"**What's New (Last {days} Days):**\n"]
        added_docs_output = []
        added_images_output = []

        # Use a set to track unique items already added to prevent duplicates
        output_tracker = set() # Store tuples like ('doc', 'doc_name') or ('image', 'image_id')

        for item in recent_items:
            # Format timestamp for display (e.g., "YYYY-MM-DD HH:MM UTC")
            ts_formatted = item['timestamp'].strftime('%Y-%m-%d %H:%M UTC')
            # Use UUID for documents in the tracker key to ensure uniqueness
            item_key = (item['type'], item.get('id'))

            if item_key not in output_tracker:
                if item['type'] == 'doc':
                    added_docs_output.append(f"- `{item['name']}` ({item['action']} {ts_formatted})")
                    output_tracker.add(item_key)
                elif item['type'] == 'image':
                    # Ensure name is fetched correctly
                    img_name = item.get('name', 'Unknown Image')
                    if img_name == 'Unknown Image' and item.get('id') in img_manager.metadata:
                        img_name = img_manager.metadata[item['id']].get('name', 'Unknown Image')

                    added_images_output.append(f"- `{img_name}` (ID: `{item['id']}`) ({item['action']} {ts_formatted})")
                    output_tracker.add(item_key)


        if added_docs_output:
            output_lines.append("**Documents:**")
            output_lines.extend(added_docs_output)
            output_lines.append("") # Add spacing

        if added_images_output:
            output_lines.append("**Images:**")
            output_lines.extend(added_images_output)

        # Combine and send, handling potential message length limits
        full_message = "\n".join(output_lines).strip() # Strip trailing whitespace
        if not full_message or full_message == f"**What's New (Last {days} Days):**": # Check if only header remains
             await interaction.followup.send(f"*database scan complete.* No relevant documents or images found in the last {days} days.")
        elif len(full_message) > 2000:
            # Simple truncation
            await interaction.followup.send(full_message[:1990] + "\n... *(message truncated)*")
        else:
            await interaction.followup.send(full_message)

    @bot.tree.command(name="pronouns", description="Set your preferred pronouns (e.g., she/her, they/them, he/him)")
    @app_commands.describe(pronouns="Your preferred pronouns")
    async def set_pronouns_command(interaction: discord.Interaction, pronouns: str):
        """Sets the user's preferred pronouns."""
        await interaction.response.defer()
        try:
            user_id = str(interaction.user.id)
            # Basic validation/sanitization could be added here if needed
            # For now, just store what the user provides. Use strip() to remove leading/trailing whitespace.
            pronouns_stripped = pronouns.strip()
            if not pronouns_stripped: # Prevent setting empty pronouns
                 await interaction.followup.send("*invalid input!* Pronouns cannot be empty.")
                 return

            success = bot.user_preferences_manager.set_pronouns(user_id, pronouns_stripped)

            if success:
                await interaction.followup.send(f"*preference updated!* Your pronouns have been set to **{pronouns_stripped}**.")
                logger.info("User %s (%s) set pronouns to: %s", interaction.user.name, user_id, pronouns_stripped)
            else:
                await interaction.followup.send("*synaptic error detected!* Failed to set your pronouns. Please try again later.")
                logger.error("Failed to set pronouns for user %s (%s)", interaction.user.name, user_id)

        except Exception as e:
            logger.error("Error in /pronouns command for user %s (%s): %s", interaction.user.name, interaction.user.id, e, exc_info=True)
            await interaction.followup.send("*neural circuit overload!* An error occurred while setting your pronouns.")


    @bot.tree.command(name="temperature", description="Set your custom temperature range")
    @app_commands.describe(
        temp_min="Minimum temperature",
        temp_base="Base temperature",
        temp_max="Maximum temperature",
    )
    async def set_temperature_command(
        interaction: discord.Interaction,
        temp_min: float | None = None,
        temp_base: float | None = None,
        temp_max: float | None = None,
    ):
        """Sets or resets the user's temperature preferences."""
        await interaction.response.defer()
        try:
            user_id = str(interaction.user.id)

            if temp_min is None and temp_base is None and temp_max is None:
                # Reset to defaults
                success = bot.user_preferences_manager.clear_temperature_settings(user_id)
                if success:
                    await interaction.followup.send(
                        "*preference updated!* Temperature settings reset to defaults."
                    )
                else:
                    await interaction.followup.send(
                        "*synaptic error detected!* Failed to reset your temperature preferences. Please try again later."
                    )
                return

            if None in (temp_min, temp_base, temp_max):
                await interaction.followup.send(
                    "*invalid input!* Provide min, base, and max values or omit all to reset."
                )
                return

            if not (0.0 <= temp_min <= temp_max <= 2.0) or not (temp_min <= temp_base <= temp_max):
                await interaction.followup.send(
                    "*invalid input!* Ensure 0.0 ≤ min ≤ base ≤ max ≤ 2.0."
                )
                return

            success = bot.user_preferences_manager.set_temperature_settings(
                user_id, temp_min, temp_base, temp_max
            )

            if success:
                await interaction.followup.send(
                    f"*preference updated!* Temperature range set to min **{temp_min}**, base **{temp_base}**, max **{temp_max}**."
                )
            else:
                await interaction.followup.send(
                    "*synaptic error detected!* Failed to set your temperature preferences. Please try again later."
                )
        except Exception as e:
            logger.error(
                "Error in /temperature command for user %s (%s): %s",
                interaction.user.name,
                interaction.user.id,
                e,
                exc_info=True,
            )
            await interaction.followup.send(
                "*neural circuit overload!* An error occurred while setting your temperature preferences."
            )


    @bot.tree.command(name="parse_channel", description="Toggle parsing of channel messages for context")
    @app_commands.describe(
        enabled="Whether to enable or disable parsing for this channel (true/false)",
        message_count="How many recent messages to parse (default: 50)"
    )
    @app_commands.choices(enabled=[
        app_commands.Choice(name="Enable", value="true"),
        app_commands.Choice(name="Disable", value="false"),
    ])
    #@app_commands.checks.has_permissions(manage_channels=True) # Only allow users who can manage channels
    async def parse_channel(interaction: discord.Interaction, enabled: str, message_count: int = 50):
        """Toggles channel message parsing and sets the message count."""
        await interaction.response.defer()
        try:
            if not interaction.channel:
                await interaction.followup.send("*neural pathway error!* This command can only be used in a server channel.")
                return

            channel_id = str(interaction.channel.id)
            enable_bool = enabled.lower() == 'true'

            if message_count <= 0 or message_count > 200: # Add a reasonable upper limit
                await interaction.followup.send("*invalid input!* Message count must be between 1 and 200.")
                return

            # Assuming a method exists in UserPreferencesManager or a dedicated ChannelSettingsManager
            # Let's use UserPreferencesManager for now, storing channel settings under a specific key
            success = bot.user_preferences_manager.set_channel_parsing_settings(channel_id, enable_bool, message_count)

            if success:
                if enable_bool:
                    await interaction.followup.send(f"*channel analysis protocol activated!* I will now parse the last **{message_count}** messages in this channel for context when mentioned.")
                else:
                    await interaction.followup.send("*channel analysis protocol deactivated!* I will no longer parse messages in this channel for context.")
                logger.info("User %s (%s) set channel parsing for channel %s to %s with count %s", interaction.user.name, interaction.user.id, channel_id, enable_bool, message_count)
            else:
                await interaction.followup.send("*synaptic error detected!* Failed to update channel parsing settings. Please try again later.")
                logger.error("Failed to set channel parsing settings for channel %s", channel_id)

        except Exception as e:
            logger.error("Error in /parse_channel command for channel %s: %s", interaction.channel.id if interaction.channel else 'N/A', e, exc_info=True)
            await interaction.followup.send("*neural circuit overload!* An error occurred while updating channel parsing settings.")

    @parse_channel.error
    async def parse_channel_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("*neural access denied!* You need the 'Manage Channels' permission to use this command.", ephemeral=True)
        else:
            logger.error("Unhandled error in /parse_channel: %s", error)
            await interaction.response.send_message("*neural circuit overload!* An unexpected error occurred.", ephemeral=True)

"""
Document management commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import re
import aiohttp
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional # Added for type hinting
from utils.helpers import split_message, check_permissions # Consolidated import & removed sanitize
from prompts.system_prompt import SYSTEM_PROMPT # Added for summarization

# Import the new function and availability flag
from managers.documents import tag_lore_in_docx, DOCX_AVAILABLE


logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all document management commands with the bot."""
    
    @bot.tree.command(name="add_info", description="Add new text to Publicia's mind for retrieval (admin only)")
    @app_commands.describe(
        name="Name of the document",
        content="Content of the document"
    )
    @app_commands.check(check_permissions)
    async def add_document(interaction: discord.Interaction, name: str, content: str):
        await interaction.response.defer()
        try:
            if not name or not content:
                await interaction.followup.send("*neural error detected!* Both name and content are required.")
                return
                
            lorebooks_path = bot.document_manager.get_lorebooks_path()
            txt_path = lorebooks_path / f"{name}.txt"
            txt_path.write_text(content, encoding='utf-8')
            
            await bot.document_manager.add_document(name, content) # Added await
            await interaction.followup.send(f"Added document: {name}\nSaved to: {txt_path}")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            await interaction.followup.send(f"Error adding document: {str(e)}")
    
    @bot.command(name="add_doc", brief="Add a new document to the knowledge base. (admin only) Usage: Publicia! add_doc \"Document Name\"")
    @commands.check(check_permissions)
    async def adddoc_prefix(ctx, *, args):
        """Add a document via prefix command with optional file attachment."""
        try:
            # Extract name from quotation marks
            match = re.match(r'"([^"]+)"', args)
            
            if not match:
                await ctx.send('*neural error detected!* Please provide a name in quotes. Example: `Publicia! add_doc "Document Name"`')
                return
                
            name = match.group(1)  # The text between quotes
            lorebooks_path = bot.document_manager.get_lorebooks_path()

            if ctx.message.attachments:
                attachment = ctx.message.attachments[0]
                if not attachment.filename.endswith('.txt'):
                    await ctx.send("Only .txt files are supported for attachments.")
                    return
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            await ctx.send("Failed to download the attachment.")
                            return
                        doc_content = await resp.text(encoding='utf-8-sig')
            else:
                # If no attachment, prompt for content
                await ctx.send("Please provide the document content (type it and send within 60 seconds) or attach a .txt file.")
                try:
                    msg = await bot.wait_for(
                        'message',
                        timeout=60.0,
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel
                    )
                    doc_content = msg.content
                except asyncio.TimeoutError:
                    await ctx.send("Timed out waiting for document content.")
                    return

            txt_path = lorebooks_path / f"{name}.txt"
            txt_path.write_text(doc_content, encoding='utf-8')
            
            await bot.document_manager.add_document(name, doc_content)
            await ctx.send(f"Added document: {name}\nSaved to: {txt_path}")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            await ctx.send(f"Error adding document: {str(e)}")

    @bot.tree.command(name="list_docs", description="List all available documents")
    async def list_documents(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            if not bot.document_manager.metadata:
                await interaction.followup.send("No documents found in the knowledge base.")
                return
                
            # Get original document names and metadata
            doc_items = []
            # Iterate through sanitized keys, get original name for display
            s_internal_list_name = bot.document_manager._sanitize_name(bot.document_manager._internal_list_doc_name)
            for s_name, meta in bot.document_manager.metadata.items():
                # Skip internal list doc
                if s_name == s_internal_list_name:
                    continue
                original_name = bot.document_manager._get_original_name(s_name)
                chunks = meta.get('chunk_count', 'N/A')
                added_raw = meta.get('added', 'Unknown')
                # Try to parse date nicely, fallback to raw string
                try:
                    added_dt = datetime.fromisoformat(added_raw)
                    added_formatted = added_dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    added_formatted = added_raw
                # Store as tuple for sorting: (original_name, display_string)
                doc_items.append((original_name.lower(), f"{original_name} - {chunks} chunks (Added: {added_formatted})"))

            # Sort alphabetically by original name (case-insensitive)
            doc_items.sort(key=lambda item: item[0])

            # Extract the display strings after sorting
            sorted_doc_strings = [item[1] for item in doc_items]

            # Create header
            header = "Available documents:"

            # Split into chunks, allowing room for code block formatting
            doc_chunks = split_message("\n".join(sorted_doc_strings), max_length=1900)  # Leave room for formatting

            for i, chunk in enumerate(doc_chunks):
                # Format each chunk as a separate code block
                formatted_chunk = f"{header if i == 0 else 'Documents (continued):'}\n```\n{chunk}\n```"
                await interaction.followup.send(formatted_chunk)
                
        except Exception as e:
            await interaction.followup.send(f"Error listing documents: {str(e)}")

    @bot.tree.command(name="remove_doc", description="Remove a document from the knowledge base (admin only)")
    @app_commands.describe(name="Name of the document to remove")
    @app_commands.check(check_permissions)
    async def remove_document(interaction: discord.Interaction, name: str):
        await interaction.response.defer()
        try:
            if not name:
                await interaction.followup.send("*neural error detected!* Please provide a document name.")
                return
                
            success = await bot.document_manager.delete_document(name) # Added await
            if success:
                await interaction.followup.send(f"Removed document: {name} \n*google docs will also need to be removed from the tracked list*")
            else:
                await interaction.followup.send(f"Document not found: {name}")
        except Exception as e:
            await interaction.followup.send(f"Error removing document: {str(e)}")

    @bot.tree.command(name="search_docs", description="Search the document knowledge base")
    @app_commands.describe(query="What to search for")
    async def search_documents(interaction: discord.Interaction, query: str):
        await interaction.response.defer()
        try:
            if not query:
                await interaction.followup.send("*neural error detected!* Please provide a search query.")
                return
                
            results = await bot.document_manager.search(query, top_k=5) # Added await
            if not results:
                await interaction.followup.send("No relevant documents found.")
                return
            
            # Create batches of results that fit within Discord's message limit
            batches = []
            current_batch = "Search results:\n"
            
            for doc_name, chunk, similarity, image_id, chunk_index, total_chunks in results:
                # Format this result
                if image_id:
                    # This is an image search result
                    image_name = bot.image_manager.metadata[image_id]['name'] if image_id in bot.image_manager.metadata else "Unknown Image"
                    result_text = f"\n**IMAGE: {image_name}** (ID: {image_id}, similarity: {similarity:.2f}):\n"
                    result_text += f"```{bot.sanitize_discord_text(chunk[:300])}...```\n"
                else:
                    result_text = f"\n**From {doc_name}** (Chunk {chunk_index}/{total_chunks}) (similarity: {similarity:.2f}):\n"
                    result_text += f"```{bot.sanitize_discord_text(chunk[:300])}...```\n"
                
                # Check if adding this result would exceed Discord's message limit
                if len(current_batch) + len(result_text) > 1900:  # Leave room for Discord's limit
                    batches.append(current_batch)
                    current_batch = "Search results (continued):\n" + result_text
                else:
                    current_batch += result_text
            
            # Add the last batch if it has content
            if current_batch and current_batch != "Search results (continued):\n":
                batches.append(current_batch)
            
            # Send each batch as a separate message
            for batch in batches:
                await interaction.followup.send(batch)
                
        except Exception as e:
            await interaction.followup.send(f"Error searching documents: {str(e)}")

    @bot.tree.command(name="add_googledoc", description="Add a Google Doc to the tracked list (admin only)")
    @app_commands.describe(
        doc_url="Google Doc URL or ID",
        name="Custom name for the document (optional)"
    )
    @app_commands.check(check_permissions)
    async def add_google_doc(interaction: discord.Interaction, doc_url: str, name: str = None):
        await interaction.response.defer()
        try:
            if not doc_url:
                await interaction.followup.send("*neural error detected!* Please provide a Google Doc URL or ID.")
                return
                
            # Extract Google Doc ID from URL if a URL is provided
            if "docs.google.com" in doc_url:
                # Extract the ID from various Google Docs URL formats
                if "/d/" in doc_url:
                    doc_id = doc_url.split("/d/")[1].split("/")[0].split("?")[0]
                elif "id=" in doc_url:
                    doc_id = doc_url.split("id=")[1].split("&")[0]
                else:
                    await interaction.followup.send("*could not extract doc id from url... is this a valid google docs link?*")
                    return
            else:
                # Assume the input is already a Doc ID
                doc_id = doc_url
            
            # If no custom name provided, try to get the document title
            if name is None:
                await interaction.followup.send("*scanning document metadata...*")
                doc_title = await bot._fetch_google_doc_title(doc_id)
                if doc_title:
                    name = doc_title
                    await interaction.followup.send(f"*document identified as: '{doc_title}'*")
            
            # Add to tracked list
            result = bot.document_manager.track_google_doc(doc_id, name)
            await interaction.followup.send(f"*synapses connecting to document ({doc_url})*\n{result}")
            
            # Download just this document instead of refreshing all
            success = await bot.refresh_single_google_doc(doc_id, name)
            
            if success:
                await interaction.followup.send("*neural pathways successfully connected!*")
            else:
                await interaction.followup.send("*neural connection established but document download failed... try refreshing later*")
        except Exception as e:
            logger.error(f"Error adding Google Doc: {e}")
            await interaction.followup.send(f"*my enhanced brain had a glitch!* couldn't add document: {str(e)}")

    @bot.tree.command(name="list_googledocs", description="List all tracked Google Docs")
    async def list_google_docs(interaction: discord.Interaction):
        await interaction.response.defer()
        tracked_file = Path(bot.document_manager.base_dir) / "tracked_google_docs.json"
        if not tracked_file.exists():
            await interaction.followup.send("*no google docs detected in my neural network...*")
            return
            
        try:
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
            
            if not tracked_docs:
                await interaction.followup.send("*my neural pathways show no connected google docs*")
                return

            # Prepare list for sorting
            doc_list_items = []
            for doc in tracked_docs:
                doc_id = doc['id']
                # Use custom name if available, otherwise generate default
                name = doc.get('custom_name') or f"googledoc_{doc_id}" # Removed .txt for sorting consistency
                doc_url = f"<https://docs.google.com/document/d/{doc_id}>"
                display_string = f"{name} - URL: {doc_url}"
                # Store tuple: (sort_key_name, display_string)
                doc_list_items.append((name.lower(), display_string))

            # Sort alphabetically by name (case-insensitive)
            doc_list_items.sort(key=lambda item: item[0])

            # Build the response string from sorted items
            response = "*accessing neural connections to google docs...*\n\n**TRACKED DOCUMENTS**\n"
            response += "\n".join([item[1] for item in doc_list_items]) # Join sorted display strings

            # Split the message to avoid Discord's 2000 character limit
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
        except Exception as e:
            logger.error(f"Error listing Google Docs: {e}")
            await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to list Google Docs.")

    @bot.tree.command(name="rename_document", description="Rename any document, Google Doc, or lorebook (admin only)")
    @app_commands.describe(
        current_name="Current name of the document to rename",
        new_name="New name for the document"
    )
    @app_commands.check(check_permissions)
    async def rename_document(interaction: discord.Interaction, current_name: str, new_name: str):
        await interaction.response.defer()
        try:
            if not current_name or not new_name:
                await interaction.followup.send("*neural error detected!* Both current name and new name are required.")
                return
                
            result = await bot.document_manager.rename_document(current_name, new_name) # Added await
            await interaction.followup.send(f"*synaptic pathways reconfiguring...*\n{result}")
        except Exception as e:
            logger.error(f"Error renaming document: {e}")
            await interaction.followup.send(f"*neural pathway error!* couldn't rename document: {str(e)}")

    @bot.tree.command(name="remove_googledoc", description="Remove a Google Doc from the tracked list (admin only)")
    @app_commands.describe(
        identifier="Google Doc ID, URL, or custom name to remove"
    )
    @app_commands.check(check_permissions)
    async def remove_google_doc(interaction: discord.Interaction, identifier: str):
        await interaction.response.defer()
        try:
            if not identifier:
                await interaction.followup.send("*neural error detected!* Please provide an identifier for the Google Doc to remove.")
                return
                
            # Path to the tracked docs file
            tracked_file = Path(bot.document_manager.base_dir) / "tracked_google_docs.json"
            if not tracked_file.exists():
                await interaction.followup.send("*no tracked google docs found in my memory banks!*")
                return
                
            # Load existing tracked docs
            with open(tracked_file, 'r') as f:
                tracked_docs = json.load(f)
            
            # Extract Google Doc ID from URL if a URL is provided
            extracted_id = None
            if "docs.google.com" in identifier:
                # Extract the ID from various Google Docs URL formats
                if "/d/" in identifier:
                    extracted_id = identifier.split("/d/")[1].split("/")[0].split("?")[0]
                elif "id=" in identifier:
                    extracted_id = identifier.split("id=")[1].split("&")[0]
            
            # Try to find and remove the doc
            removed = False
            for i, doc in enumerate(tracked_docs):
                # Priority: 1. Direct ID match, 2. Custom name match, 3. Extracted URL ID match
                if doc['id'] == identifier or \
                    (doc.get('custom_name') and doc.get('custom_name') == identifier) or \
                    (extracted_id and doc['id'] == extracted_id):
                    removed_doc = tracked_docs.pop(i)
                    removed = True
                    break
            
            if not removed:
                await interaction.followup.send(f"*hmm, i couldn't find a document matching '{identifier}' in my neural network*")
                return
            
            # Save updated list
            with open(tracked_file, 'w') as f:
                json.dump(tracked_docs, f)
                
            # Get the original URL and document name for feedback
            doc_id = removed_doc['id']
            doc_url = f"https://docs.google.com/document/d/{doc_id}"
            doc_name = removed_doc.get('custom_name') or f"googledoc_{doc_id}"

            local_file_name = doc_name
            if not local_file_name.endswith('.txt'):
                local_file_name += '.txt'

            # Remove local file if it exists in the document manager
            file_removed = False
            logger.info(f"Attempting to delete local document associated with Google Doc: '{local_file_name}'")
            try:
                # Use the original name (local_file_name) which might be custom or ID-based
                success = await bot.document_manager.delete_document(local_file_name)
                if success:
                    logger.info(f"Successfully deleted local document file '{local_file_name}' via DocumentManager.")
                    file_removed = True
                else:
                    # It's possible the doc was tracked but never successfully downloaded/added
                    logger.warning(f"Google Doc '{doc_name}' removed from tracking, but corresponding local document '{local_file_name}' not found in DocumentManager for deletion.")
                    # Send a less alarming message in this case, as it might be expected
                    # await interaction.followup.send(f"Note: Google Doc tracking removed, but local file '{local_file_name}' was not found in the document manager.")
            except Exception as e:
                logger.error(f"Error attempting to delete local document '{local_file_name}' during Google Doc removal: {e}")
                # Inform user about the error, but continue with tracking removal confirmation
                await interaction.followup.send(f"*neural warning!* Error removing associated local document file '{local_file_name}': {str(e)}")

            response = f"*I've surgically removed the neural connection to {doc_name}*, it will no longer be tracked\n*url: {doc_url}*"
            if file_removed:
                response += f"\n*and removed the associated local document file ({local_file_name}) from my knowledge base.*"
            else:
                 response += f"\n*(Associated local file '{local_file_name}' was not found in the knowledge base to remove)*"

            for chunk in split_message(response):
                await interaction.followup.send(chunk)
                
        except Exception as e:
            logger.error(f"Error removing Google Doc: {e}")
            await interaction.followup.send(f"*my enhanced brain experienced an error!* couldn't remove document: {str(e)}")
    
    @bot.tree.command(name="list_files", description="List files that Publicia can upload")
    @app_commands.describe(
        file_type="Type of files to list",
        search_term="Optional search term to filter files"
    )
    @app_commands.choices(file_type=[
        app_commands.Choice(name="Documents", value="documents"),
        app_commands.Choice(name="Lorebooks", value="lorebooks"),
        app_commands.Choice(name="All", value="all")
    ])
    async def list_files(interaction: discord.Interaction, file_type: str, search_term: str = None):
        await interaction.response.defer()
        try:
            files = []
            
            # Helper function to list files in a directory
            def get_files_from_dir(dir_path, prefix=""):
                results = []
                try:
                    for item in Path(dir_path).glob("*"):
                        if item.is_file() and not item.name.startswith(".") and not item.name.endswith(".pkl") and not item.name.endswith(".json"):
                            if search_term is None or search_term.lower() in item.name.lower():
                                results.append(f"{prefix}{item.name}")
                except Exception as e:
                    logger.error(f"Error listing files in {dir_path}: {e}")
                return results
            
            # Get files based on type
            if file_type == "documents" or file_type == "all":
                docs = get_files_from_dir(bot.document_manager.base_dir, "[Document] ")
                files.extend(docs)
            
            # Lorebooks are also documents, so no need to check for "all" again
            if file_type == "lorebooks":
                lorebooks = get_files_from_dir(bot.document_manager.get_lorebooks_path(), "[Lorebook] ")
                files.extend(lorebooks)
            
            # Send response
            if not files:
                if search_term:
                    await interaction.followup.send(f"*neural search complete!* I couldn't find any files matching '{search_term}' in the selected storage areas.")
                else:
                    await interaction.followup.send("*neural storage scan complete!* I don't have any files of this type in my memory banks.")
                return
            
            # Create formatted response
            response = "*accessing neural storage banks...*\n\n**AVAILABLE FILES**\n\n"
            for i, file_name in enumerate(sorted(files)):
                response += f"{i+1}. {file_name}\n"
                
            # Add usage instructions
            response += "\n*To retrieve a file, use the `/retrieve_file` command with the file name (without type prefix)*"
            
            # Split if necessary
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
        
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while trying to list files.")

    @bot.tree.command(name="retrieve_file", description="Retrieve a file that Publicia has saved")
    @app_commands.describe(
        file_name="Name of the file to retrieve",
        file_type="Type of the file"
    )
    @app_commands.choices(file_type=[
        app_commands.Choice(name="Document", value="document"),
        app_commands.Choice(name="Lorebook", value="lorebook")
    ])
    async def retrieve_file(interaction: discord.Interaction, file_name: str, file_type: str):
        await interaction.response.defer()
        try:
            # Determine the directory based on file type
            if file_type == "document":
                base_dir = bot.document_manager.base_dir
            elif file_type == "lorebook":
                base_dir = bot.document_manager.get_lorebooks_path()
            else:
                await interaction.followup.send("*neural error detected!* Invalid file type. Please select 'Document' or 'Lorebook'.")
                return
            
            # Try exact match first
            path = Path(base_dir) / file_name
            
            # If not found, try adding .txt extension
            if not path.exists() and not file_name.endswith('.txt'):
                path = Path(base_dir) / f"{file_name}.txt"
            
            # If still not found, try case-insensitive search
            if not path.exists():
                found = False
                try:
                    for item in Path(base_dir).glob("*"):
                        if item.is_file() and (item.name.lower() == file_name.lower() or 
                        item.name.lower() == f"{file_name.lower()}.txt"):
                            path = item
                            found = True
                            break
                except Exception as e:
                    logger.error(f"Error searching for file: {e}")
                
                if not found:
                    await interaction.followup.send(f"*neural error detected!* Could not find file '{file_name}' in {file_type} storage.")
                    return
            
            # Check if the file exists and is a file
            if not path.exists() or not path.is_file():
                await interaction.followup.send(f"*neural error detected!* Could not find file: {file_name}")
                return
            
            # Send the file
            file = discord.File(path, filename=path.name)
            await interaction.followup.send(
                content=f"*neural pathways accessing storage...* Here is the requested {file_type}: `{path.name}`",
                file=file
            )
        
        except Exception as e:
            logger.error(f"Error retrieving file: {e}")
            await interaction.followup.send("*neural circuit overload!* An error occurred while trying to retrieve the file.")

    @bot.tree.command(name="refresh_docs", description="Manually refresh all tracked Google Docs (admin only)")
    @app_commands.check(check_permissions)
    async def refresh_docs(interaction: discord.Interaction):
        """Manually trigger the refresh process for all tracked Google Docs."""
        await interaction.response.defer()
        try:
            await interaction.followup.send("*neural pathways activating... initiating Google Docs refresh sequence...*")
            
            # Call the refresh method from the bot instance
            await bot.refresh_google_docs()
            
            await interaction.followup.send("*neural synchronization complete!* Google Docs refresh finished.")
        except Exception as e:
            logger.error(f"Error during manual Google Docs refresh: {e}")
            await interaction.followup.send(f"*neural circuit overload!* An error occurred during refresh: {str(e)}")

    @bot.tree.command(name="archive_channel", description="Archive messages from a Discord channel as a document (admin only)")
    @app_commands.describe(
        channel="The channel to archive messages from",
        message_limit="Maximum number of messages to fetch (0 for all, default: 1000, max: 10000)",
        document_name="Name for the saved document (default: channel name)",
        include_bots="Whether to include messages from bots (default: True)",
        include_attachments="Whether to include attachment URLs (default: True)"
    )
    @app_commands.check(check_permissions)
    async def archive_channel(
        interaction: discord.Interaction, 
        channel: discord.TextChannel, 
        message_limit: int = 1000,
        document_name: str = None,
        include_bots: bool = True,
        include_attachments: bool = True
    ):
        await interaction.response.defer()
        try:
            # Validate inputs
            if message_limit < 0:
                await interaction.followup.send("*neural error detected!* Message limit must be non-negative.")
                return
                
            # Enforce maximum message limit
            MAX_MESSAGES = 10000  # Maximum number of messages that can be archived
            if message_limit == 0 or message_limit > MAX_MESSAGES:
                message_limit = MAX_MESSAGES
                await interaction.followup.send(f"*neural caution!* Message limit capped at {MAX_MESSAGES} messages for performance reasons.")
                
            # Set default document name if not provided
            if not document_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                document_name = f"channel_archive_{channel.name}_{timestamp}"
                
            # Ensure document name has .txt extension
            if not document_name.endswith('.txt'):
                document_name += '.txt'
                
            # Send initial status message
            status_message = await interaction.followup.send("*neural archiving process initiated...* scanning channel messages")
            
            # Set up for careful fetching to avoid rate limits
            messages = []
            batch_size = 100  # Fetch in batches of 100 messages
            progress_interval = 500  # Update progress every 500 messages
            
            # Track timestamps for updates
            last_update_time = time.time()
            update_interval = 2.0  # Update status every 2 seconds
            
            # Update status
            await status_message.edit(content="*neural pathways connecting to channel history...* retrieving messages")
            
            # Function to fetch messages in smaller batches with delays
            async def fetch_messages_carefully(limit=None):
                collected = []
                last_id = None
                remaining = limit
                
                while True:
                    # Calculate batch size for this iteration
                    current_batch = min(batch_size, remaining if remaining is not None else batch_size)
                    if current_batch <= 0:
                        break
                        
                    # Fetch a batch of messages
                    kwargs = {'limit': current_batch}
                    if last_id:
                        kwargs['before'] = discord.Object(id=last_id)
                        
                    batch = []
                    async for message in channel.history(**kwargs):
                        # Skip bot messages if not including them
                        if not include_bots and message.author.bot:
                            continue
                            
                        batch.append(message)
                        
                    # If we got no messages, we're done
                    if not batch:
                        break
                        
                    # Process the batch
                    collected.extend(batch)
                    
                    # Update progress periodically
                    current_time = time.time()
                    if current_time - last_update_time > update_interval:
                        await status_message.edit(content=f"*neural pathways collecting... fetched {len(collected)} messages so far...*")
                        last_update_time = current_time
                    
                    # Update last_id for next batch
                    last_id = batch[-1].id
                    
                    # Brief delay between batches to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                    # If we got fewer messages than requested, we're done
                    if len(batch) < current_batch:
                        break
                        
                    # If we've reached the max limit, stop
                    if limit is not None and len(collected) >= limit:
                        collected = collected[:limit]  # Ensure we don't exceed the limit
                        break
                
                return collected
                
            # Fetch messages
            messages = await fetch_messages_carefully(message_limit)
            
            if not messages:
                await status_message.edit(content="*neural archiving complete!* No messages found in the channel or all were filtered out.")
                return
            
            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda m: m.created_at)
            
            # Update status
            await status_message.edit(content=f"*neural formatting process initiated...* formatting {len(messages)} messages")
            
            # Initialize counters
            message_count = 0
            attachment_count = 0
            reaction_count = 0
            embed_count = 0
            
            # Initialize content with header
            content = f"# DISCORD CHANNEL ARCHIVE: {channel.name}\n"
            content += f"# Server: {interaction.guild.name}\n"
            content += f"# Archived by: {interaction.user.name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"# Message count: {len(messages)}\n"
            content += f"# Bot messages: {'Included' if include_bots else 'Excluded'}\n"
            content += f"# Attachments: {'Included' if include_attachments else 'Excluded'}\n\n"
            
            # Check if we need to split into multiple documents (if over 1MB)
            max_doc_size = 1000000  # 1MB limit
            current_doc_size = len(content.encode('utf-8'))
            current_doc_content = content
            doc_count = 1
            documents = []
            
            # Process messages
            for i, message in enumerate(messages):
                # Show progress for large archives
                if i % 500 == 0 and i > 0:
                    await status_message.edit(content=f"*neural formatting in progress...* processed {i}/{len(messages)} messages")
                
                message_count += 1
                
                # Format timestamp
                timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
                
                # Format author with roles if available
                author_name = message.author.display_name
                author_roles = ""
                if hasattr(message.author, "roles") and len(message.author.roles) > 1:  # Skip @everyone role
                    role_names = [role.name for role in message.author.roles if role.name != "@everyone"]
                    if role_names:
                        author_roles = f" [{', '.join(role_names)}]"
                
                # Add message header
                message_content = f"\n[{timestamp}] {author_name}{author_roles}:\n"
                
                # Add message content
                if message.content:
                    # Process mentions in the content
                    msg_content = message.content
                    
                    # Replace user mentions with names
                    for mention in message.mentions:
                        mention_name = mention.display_name
                        msg_content = msg_content.replace(f'<@{mention.id}>', f'@{mention_name}')
                        msg_content = msg_content.replace(f'<@!{mention.id}>', f'@{mention_name}')
                    
                    # Replace channel mentions with names
                    for channel_id in re.findall(r'<#(\d+)>', msg_content):
                        try:
                            mentioned_channel = interaction.guild.get_channel(int(channel_id))
                            if mentioned_channel:
                                msg_content = msg_content.replace(f'<#{channel_id}>', f'#{mentioned_channel.name}')
                        except:
                            pass
                    
                    # Replace role mentions with names
                    for role_id in re.findall(r'<@&(\d+)>', msg_content):
                        try:
                            mentioned_role = interaction.guild.get_role(int(role_id))
                            if mentioned_role:
                                msg_content = msg_content.replace(f'<@&{role_id}>', f'@{mentioned_role.name}')
                        except:
                            pass
                    
                    # Add the processed content with indentation
                    for line in msg_content.split('\n'):
                        message_content += f"    {line}\n"
                
                # Add attachments
                if message.attachments and include_attachments:
                    message_content += "    [Attachments]\n"
                    for attachment in message.attachments:
                        message_content += f"    • {attachment.filename} - {attachment.url}\n"
                        attachment_count += 1
                
                # Add embeds
                if message.embeds:
                    message_content += "    [Embeds]\n"
                    for embed in message.embeds:
                        embed_count += 1
                        if embed.title:
                            message_content += f"    • Title: {embed.title}\n"
                        if embed.description:
                            message_content += f"      Description: {embed.description[:100]}{'...' if len(embed.description) > 100 else ''}\n"
                
                # Add reactions
                if message.reactions:
                    reaction_str = "    [Reactions] "
                    for reaction in message.reactions:
                        reaction_count += 1
                        emoji = reaction.emoji if isinstance(reaction.emoji, str) else reaction.emoji.name
                        reaction_str += f"{emoji} ({reaction.count}) "
                    message_content += reaction_str + "\n"
                
                message_size = len(message_content.encode('utf-8'))
                
                # Check if adding this message would exceed the limit
                if current_doc_size + message_size > max_doc_size:
                    # Add footer with statistics for this part
                    part_footer = f"\n\n# ARCHIVE SUMMARY (PART {doc_count})\n"
                    part_footer += f"# Total messages in this part: {message_count}\n"
                    part_footer += f"# Attachments: {attachment_count}\n"
                    part_footer += f"# Reactions: {reaction_count}\n"
                    part_footer += f"# Embeds: {embed_count}\n"
                    part_footer += f"# Continued in part {doc_count + 1}\n"
                    
                    current_doc_content += part_footer
                    
                    # Save current document and start a new one
                    doc_filename = document_name.replace('.txt', f'_part{doc_count}.txt')
                    documents.append((doc_filename, current_doc_content))
                    doc_count += 1
                    
                    # Reset counters for next part
                    message_count_so_far = message_count
                    attachment_count_so_far = attachment_count
                    reaction_count_so_far = reaction_count
                    embed_count_so_far = embed_count
                    
                    # Start new document with header
                    current_doc_content = f"# DISCORD CHANNEL ARCHIVE: {channel.name} (PART {doc_count})\n"
                    current_doc_content += f"# Server: {interaction.guild.name}\n"
                    current_doc_content += f"# Archived by: {interaction.user.name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    current_doc_content += f"# Continued from part {doc_count - 1}\n"
                    current_doc_content += f"# Bot messages: {'Included' if include_bots else 'Excluded'}\n"
                    current_doc_content += f"# Attachments: {'Included' if include_attachments else 'Excluded'}\n\n"
                    
                    current_doc_size = len(current_doc_content.encode('utf-8'))
                
                # Add message to current document
                current_doc_content += message_content
                current_doc_size += message_size
            
            # Add footer with statistics for the last part
            footer = f"\n\n# ARCHIVE SUMMARY"
            if doc_count > 1:
                footer += f" (PART {doc_count})"
            footer += f"\n# Total messages: {message_count}\n"
            footer += f"# Attachments: {attachment_count}\n"
            footer += f"# Reactions: {reaction_count}\n"
            footer += f"# Embeds: {embed_count}\n"
            
            current_doc_content += footer
            
            # Add the final document
            final_filename = document_name if doc_count == 1 else document_name.replace('.txt', f'_part{doc_count}.txt')
            documents.append((final_filename, current_doc_content))
            
            # Update status
            await status_message.edit(content=f"*neural storage process initiated...* saving {len(messages)} messages to document system")
            
            # Save documents
            lorebooks_path = bot.document_manager.get_lorebooks_path()
            file_paths = []
            
            for doc_name, doc_content in documents:
                # Ensure the document name is clean
                clean_doc_name = doc_name.strip()
                if not clean_doc_name.endswith('.txt'):
                    clean_doc_name += '.txt'
                    
                # Save as a document file
                txt_path = lorebooks_path / clean_doc_name
                txt_path.write_text(doc_content, encoding='utf-8')
                file_paths.append(txt_path)
                
                # Add to document manager
                await bot.document_manager.add_document(clean_doc_name, doc_content)
            
            # Create downloadable files
            if len(documents) == 1:
                # Single file case
                file = discord.File(file_paths[0], filename=documents[0][0])
                
                # Send completion message with file
                await interaction.followup.send(
                    content=f"*neural archiving complete!* Successfully archived {len(messages)} messages from #{channel.name}.\n\n"
                            f"• Document name: `{documents[0][0]}`\n"
                            f"• Messages: {message_count}\n"
                            f"• Attachments: {attachment_count}\n"
                            f"• Reactions: {reaction_count}\n"
                            f"• Embeds: {embed_count}\n\n"
                            f"The archive has been added to my knowledge base and is also available as a downloadable file below:",
                    file=file
                )
            else:
                # Multiple files case - only attach the first file
                file = discord.File(file_paths[0], filename=documents[0][0])
                
                doc_names = ', '.join([f"`{doc[0]}`" for doc in documents])
                
                # Send completion message with first file
                await interaction.followup.send(
                    content=f"*neural archiving complete!* Successfully archived {len(messages)} messages from #{channel.name}.\n\n"
                            f"• Document parts: {len(documents)}\n"
                            f"• Document names: {doc_names}\n"
                            f"• Messages: {message_count}\n"
                            f"• Attachments: {attachment_count}\n"
                            f"• Reactions: {reaction_count}\n"
                            f"• Embeds: {embed_count}\n\n"
                            f"The archive has been added to my knowledge base. The first part is attached below. "
                            f"Use the `/retrieve_file` command to download other parts.",
                    file=file
                )
            
        except Exception as e:
            logger.error(f"Error archiving channel: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await interaction.followup.send(f"*neural circuit overload!* An error occurred while archiving the channel: {str(e)}")

    @bot.tree.command(name="summarize_doc", description="Generate a summary of a document")
    @app_commands.describe(document_name="The name of the document to summarize")
    async def summarize_document(interaction: discord.Interaction, document_name: str):
        """Generates a summary of the specified document using an LLM."""
        await interaction.response.defer()
        try:
            if not document_name:
                await interaction.followup.send("*neural error detected!* Please provide a document name.")
                return

            # --- Find and read the document content ---
            doc_content = None
            doc_mgr = bot.document_manager
            base_dir = doc_mgr.base_dir
            lorebooks_dir = doc_mgr.get_lorebooks_path()
            possible_paths = [
                base_dir / document_name,
                lorebooks_dir / document_name,
                base_dir / f"{document_name}.txt",
                lorebooks_dir / f"{document_name}.txt"
            ]

            found_path = None
            for path in possible_paths:
                if path.exists() and path.is_file():
                    found_path = path
                    break

            if found_path:
                logger.info(f"Found document file for summarization: {found_path}")
                try:
                    doc_content = found_path.read_text(encoding='utf-8-sig')
                except Exception as read_err:
                    logger.error(f"Error reading document file {found_path}: {read_err}")
                    await interaction.followup.send(f"*neural error!* Failed to read the document file: {read_err}")
                    return
                else:
                    # Fallback: Concatenate chunks using sanitized name lookup
                    logger.info(f"Document file not found for '{document_name}'. Attempting to concatenate chunks via DocumentManager.")
                    s_name = doc_mgr._get_sanitized_name_from_original(document_name)
                    if s_name and s_name in doc_mgr.chunks:
                        doc_content = "\n\n".join(doc_mgr.chunks[s_name])
                        logger.info(f"Concatenated {len(doc_mgr.chunks[s_name])} chunks for '{document_name}' (sanitized: {s_name}).")
                    else:
                        # Try adding .txt extension to original name before sanitizing
                        s_name_txt = doc_mgr._get_sanitized_name_from_original(f"{document_name}.txt")
                        if s_name_txt and s_name_txt in doc_mgr.chunks:
                             doc_content = "\n\n".join(doc_mgr.chunks[s_name_txt])
                             logger.info(f"Concatenated {len(doc_mgr.chunks[s_name_txt])} chunks for '{document_name}.txt' (sanitized: {s_name_txt}).")
                        else:
                            await interaction.followup.send(f"*neural error!* Document '{document_name}' not found in my knowledge base or associated files.")
                            return

            if not doc_content or not doc_content.strip():
                await interaction.followup.send(f"*neural analysis complete!* The document '{document_name}' appears to be empty.")
                return

            # --- Prepare for LLM call ---
            status_message = await interaction.followup.send(f"*neural pathways activating... preparing summary for '{document_name}'...*")

            # Use user's preferred model or default
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id),
                default_model=bot.config.LLM_MODEL
            )
            # Select a capable model (prefer larger context if possible, but use preferred)
            # For summarization, a model with good comprehension is needed.
            summary_model = preferred_model # Use user's preference for consistency

            # Construct the prompt
            # Using a simplified system prompt for summarization task
            summary_system_prompt = "You are an AI assistant tasked with summarizing documents accurately and concisely. Respond only with the summary."
            prompt_messages = [
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": f"Please summarize the following document titled '{document_name}':\n\n<document_content>\n{doc_content}\n</document_content>\n\nProvide a concise summary of the key information."}
            ]

            await status_message.edit(content=f"*neural core processing... generating summary for '{document_name}' using {summary_model}...*")

            # --- Call the LLM ---
            # Use a moderate temperature for summarization
            temperature = 0.1
            completion, actual_model = await bot._try_ai_completion(
                summary_model,
                prompt_messages,
                temperature=temperature
            )

            # --- Process and send the response ---
            if completion and completion.get('choices') and len(completion['choices']) > 0:
                if 'message' in completion['choices'][0] and 'content' in completion['choices'][0]['message']:
                    summary_text = completion['choices'][0]['message']['content']
                    response_header = f"**Summary for Document: {document_name}** (Generated by {actual_model})\n\n"
                    
                    # Delete the status message and send the final summary
                    try:
                        await status_message.delete()
                    except discord.NotFound:
                        pass # Message already deleted or interaction expired

                    await bot.send_split_message(
                        interaction.channel,
                        response_header + summary_text,
                        user_id=str(interaction.user.id) # Pass user_id for potential history logging if needed later
                    )
                else:
                    logger.error(f"Unexpected summary response structure: {completion}")
                    await status_message.edit(content="*neural circuit overload!* I received an unexpected response structure while summarizing.")
            else:
                await status_message.edit(content=f"*synaptic failure detected!* I apologize, but I couldn't generate a summary for '{document_name}' right now.")

        except Exception as e:
            logger.error(f"Error summarizing document '{document_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                # Try to edit the status message first
                if 'status_message' in locals() and status_message:
                    await status_message.edit(content=f"*neural circuit overload!* An error occurred while summarizing: {str(e)}")
                else:
                    await interaction.followup.send(f"*neural circuit overload!* An error occurred while summarizing: {str(e)}")
            except discord.NotFound:
                 await interaction.followup.send(f"*neural circuit overload!* An error occurred while summarizing: {str(e)}")
            except Exception as send_err:
                 logger.error(f"Failed to send error message for summarize_doc: {send_err}")

    @bot.tree.command(name="view_chunk", description="View the content of a specific document chunk")
    @app_commands.describe(
        document_name="The name of the document",
        chunk_index="The index of the chunk to view (starting from 1)",
        contextualized="View the contextualized version used for embeddings (default: False)"
    )
    async def view_chunk(interaction: discord.Interaction, document_name: str, chunk_index: int, contextualized: bool = False):
        """Displays the content of a specific chunk from a document."""
        await interaction.response.defer()
        try:
            if not document_name:
                await interaction.followup.send("*neural error detected!* Please provide a document name.")
                return
            if chunk_index <= 0:
                await interaction.followup.send("*neural error detected!* Chunk index must be 1 or greater.")
                return

            doc_mgr = bot.document_manager
            chunks_list = None
            total_chunks = 0
            actual_doc_name_used = document_name # Keep track of which name variation worked

            # Find the sanitized name corresponding to the user input
            s_name = doc_mgr._get_sanitized_name_from_original(document_name)
            if not s_name:
                 # Try adding .txt if original name didn't work
                 s_name = doc_mgr._get_sanitized_name_from_original(f"{document_name}.txt")
                 if s_name:
                     actual_doc_name_used = f"{document_name}.txt" # Update the name that worked

            if not s_name:
                await interaction.followup.send(f"*neural error!* Document '{document_name}' not found in my knowledge base.")
                return

            # Determine which chunk list to use based on 'contextualized' flag
            chunk_source_dict = doc_mgr.contextualized_chunks if contextualized else doc_mgr.chunks
            chunk_type = "Contextualized" if contextualized else "Original"

            if s_name in chunk_source_dict:
                chunks_list = chunk_source_dict[s_name]
                total_chunks = len(chunks_list)
            elif contextualized and s_name in doc_mgr.chunks:
                 # Fallback for contextualized request if only original exists
                 logger.warning(f"Contextualized chunk requested for '{document_name}' (sanitized: {s_name}), but not found. Falling back to original.")
                 chunks_list = doc_mgr.chunks[s_name]
                 total_chunks = len(chunks_list)
                 chunk_type = "Original (Fallback)" # Indicate fallback
            else:
                 # This case should be rare if s_name was found via _get_sanitized_name_from_original
                 await interaction.followup.send(f"*neural error!* Could not find chunk data for document '{document_name}' (sanitized: {s_name}).")
                 return

            # Convert 1-based index to 0-based
            zero_based_index = chunk_index - 1

            if not chunks_list or zero_based_index < 0 or zero_based_index >= total_chunks:
                await interaction.followup.send(f"*neural error!* Invalid chunk index {chunk_index}. Document '{actual_doc_name_used}' has {total_chunks} chunks.")
                return

            # Get the chunk content
            chunk_content = chunks_list[zero_based_index]

            # Use the original name stored in metadata for the header
            original_name_for_display = doc_mgr._get_original_name(s_name)
            response_header = f"**{chunk_type} Chunk {chunk_index}/{total_chunks} from Document: {original_name_for_display}**"

            # Sanitize the raw chunk content *before* splitting
            # Assuming bot has sanitize_discord_text method, otherwise remove/adjust
            sanitized_content = chunk_content # Default if sanitize method doesn't exist or causes issues
            if hasattr(bot, 'sanitize_discord_text') and callable(bot.sanitize_discord_text):
                 sanitized_content = bot.sanitize_discord_text(chunk_content)
            else:
                 logger.warning("bot.sanitize_discord_text method not found or not callable.")


            # Split the sanitized content (leave room for code block backticks and newlines)
            # Max length for content inside code block: 2000 - len("```\n\n```") = 1992. Use 1980 for safety.
            content_parts = split_message(sanitized_content, max_length=1980)

            # Send the header first using followup.send()
            # This resolves the initial deferral.
            await interaction.followup.send(response_header)

            # Send each content part in its own code block using followup.send()
            if not content_parts:
                 await interaction.followup.send("```\n(Chunk content is empty)\n```")
            else:
                for part in content_parts:
                    # Ensure part is not empty before sending
                    if part and part.strip():
                        await interaction.followup.send(f"```\n{part}\n```")
                    else:
                        # Send placeholder if part is empty/whitespace after split
                        await interaction.followup.send("```\n(Empty part)\n```")


        except Exception as e:
            logger.error(f"Error viewing chunk {chunk_index} from '{document_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Always use followup.send for errors after deferral
            error_message = f"*neural circuit overload!* An error occurred while viewing the chunk: {str(e)}"
            try:
                # Check if interaction is still valid before sending followup
                if interaction.response.is_done():
                     await interaction.followup.send(error_message)
                else:
                     # If somehow defer wasn't called or failed, edit original (less likely)
                     await interaction.edit_original_response(content=error_message)
            except discord.NotFound:
                 logger.warning(f"Interaction or original message not found when trying to send error for view_chunk.")
                 # Fallback: Try sending to channel if interaction failed
                 try:
                     await interaction.channel.send(f"{interaction.user.mention} {error_message}")
                 except Exception as final_err:
                     logger.error(f"Failed to send view_chunk error message via channel: {final_err}")
                 except Exception as send_err:
                      logger.error(f"Failed to send error message for view_chunk via followup: {send_err}")

    @bot.tree.command(name="process_docx_lore", description="Process a .docx file to tag specific colored text with XML tags")
    @app_commands.describe(
        docx_file="The .docx file to process",
        output_filename="Optional name for the output .txt file (default: [input_name]_processed.txt)"
    )
    @app_commands.check(check_permissions)
    async def process_docx_lore(interaction: discord.Interaction, docx_file: discord.Attachment, output_filename: Optional[str] = None):
        """Processes an uploaded .docx file to add XML tags based on text color."""
        await interaction.response.defer()
        try:
            # Check if python-docx is available
            if not DOCX_AVAILABLE:
                await interaction.followup.send("*neural error!* The required 'python-docx' library is not installed on the bot's system. This command cannot function.")
                return

            # Check if the attachment is a .docx file
            if not docx_file.filename.lower().endswith('.docx'):
                await interaction.followup.send("*neural error detected!* Please upload a valid .docx file.")
                return

            # Create a temporary directory if it doesn't exist
            temp_dir = Path("./temp_files")
            temp_dir.mkdir(exist_ok=True)
            
            # Define input and output paths
            input_path = temp_dir / docx_file.filename
            
            # Determine output filename
            if output_filename:
                # Ensure it ends with .txt
                if not output_filename.lower().endswith('.txt'):
                    output_filename += '.txt'
            else:
                # Default: input filename + _processed.txt
                base_name = Path(docx_file.filename).stem
                output_filename = f"{base_name}_processed.txt"
            
            output_path = temp_dir / output_filename

            # Download the attached .docx file
            await interaction.followup.send(f"*neural pathways receiving file...* downloading `{docx_file.filename}`...")
            await docx_file.save(input_path)
            
            await interaction.followup.send(f"*neural core processing...* analyzing `{docx_file.filename}` for lore tags...")

            # Process the .docx file using the function from DocumentManager
            processed_content = tag_lore_in_docx(str(input_path))

            if processed_content is None:
                await interaction.followup.send(f"*neural circuit overload!* Failed to process the document `{docx_file.filename}`. Check bot logs for details.")
                # Clean up downloaded file
                if input_path.exists():
                    input_path.unlink()
                return

            # Save the processed content to the output .txt file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
            except Exception as write_err:
                 logger.error(f"Error writing processed content to {output_path}: {write_err}")
                 await interaction.followup.send(f"*neural storage error!* Failed to save the processed file: {write_err}")
                 # Clean up files
                 if input_path.exists(): input_path.unlink()
                 if output_path.exists(): output_path.unlink()
                 return


            # Send the processed file back to the user
            if output_path.exists():
                await interaction.followup.send(
                    content=f"*neural processing complete!* Here is the processed file `{output_filename}` with lore tags:",
                    file=discord.File(output_path, filename=output_filename)
                )
            else:
                 # This case should be rare if writing didn't throw an error, but safety check
                 await interaction.followup.send(f"*neural error!* Processed file `{output_filename}` could not be found after saving.")


            # Clean up temporary files
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()

        except Exception as e:
            logger.error(f"Error processing docx lore for '{docx_file.filename}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            await interaction.followup.send(f"*neural circuit overload!* An unexpected error occurred: {str(e)}")
            # Attempt cleanup even on error
            try:
                if 'input_path' in locals() and input_path.exists(): input_path.unlink()
                if 'output_path' in locals() and output_path.exists(): output_path.unlink()
            except Exception as cleanup_err:
                logger.error(f"Error during cleanup after process_docx_lore error: {cleanup_err}")

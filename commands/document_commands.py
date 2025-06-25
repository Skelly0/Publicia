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
import uuid # Added for UUID validation
from datetime import datetime
from pathlib import Path
from typing import Optional # Added for type hinting
from utils.helpers import split_message, check_permissions, sanitize_filename # Consolidated import & removed sanitize
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
                await interaction.followup.send("*neural error detected!* Both original name and content are required.")
                return
            
            # The DocumentManager's add_document now handles UUID generation and saving the .txt file.
            # It returns the UUID of the new document.
            doc_uuid = await bot.document_manager.add_document(original_name=name, content=content) 
            
            if doc_uuid:
                await interaction.followup.send(f"Document '{name}' added successfully with UUID: `{doc_uuid}`.")
            else:
                await interaction.followup.send(f"Failed to add document '{name}'. Check logs for details.")
            
        except Exception as e:
            logger.error(f"Error adding document via /add_info: {e}", exc_info=True)
            await interaction.followup.send(f"Error adding document: {str(e)}")
    
    @bot.command(name="add_doc", brief="Add a new document to the knowledge base. (admin only) Usage: Publicia! add_doc \"Document Name\"")
    @commands.check(check_permissions)
    async def adddoc_prefix(ctx, *, args: Optional[str] = None):
        """Add a document via prefix command with optional file attachment."""
        try:
            # If no args and no attachments, there's nothing to do.
            if args is None and not ctx.message.attachments:
                await ctx.send('*neural error detected!* Please provide a document name in quotes, attach a file, or both.')
                return

            name = None
            if args:
                # Extract name from quotation marks if args are provided
                match = re.match(r'"([^"]+)"', args)
                if match:
                    name = match.group(1)
                else:
                    # If args are provided but not in quotes, it's an error
                    await ctx.send('*neural error detected!* If providing a name, it must be in quotes. Example: `Publicia! add_doc "My Doc"`')
                    return
            
            doc_content = None
            if ctx.message.attachments:
                attachment = ctx.message.attachments[0]
                # If no name was provided in args, use the attachment's filename
                if name is None:
                    name = Path(attachment.filename).stem # Use filename without extension

                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                        if resp.status != 200:
                            await ctx.send("Failed to download the attachment.")
                            return
                        try:
                            doc_content = await resp.text(encoding='utf-8-sig')
                        except UnicodeDecodeError:
                            logger.warning(f"Failed to decode attachment {attachment.filename} as utf-8-sig, trying plain utf-8.")
                            doc_content = await resp.text(encoding='utf-8')
            else:
                # This block now only runs if there are args but no attachment
                if name is None: # Should not happen due to earlier checks, but for safety
                    await ctx.send("An unknown error occurred: No name could be determined.")
                    return

                await ctx.send("Please provide the document content (type it and send within 60 seconds).")
                try:
                    msg = await bot.wait_for(
                        'message',
                        timeout=60.0,
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel and not m.attachments
                    )
                    doc_content = msg.content
                except asyncio.TimeoutError:
                    await ctx.send("Timed out waiting for document content.")
                    return

            # Final check for content
            if doc_content is None:
                await ctx.send("Could not find any content to add.")
                return

            # DocumentManager's add_document now handles UUID generation and saving.
            doc_uuid = await bot.document_manager.add_document(original_name=name, content=doc_content)
            
            if doc_uuid:
                await ctx.send(f"Document '{name}' added successfully with UUID: `{doc_uuid}`.")
            else:
                await ctx.send(f"Failed to add document '{name}'. Check logs for details.")
            
        except Exception as e:
            logger.error(f"Error adding document via !add_doc: {e}", exc_info=True)
            await ctx.send(f"Error adding document: {str(e)}")

    @bot.tree.command(name="list_docs", description="List all available documents")
    async def list_documents(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            if not bot.document_manager.metadata:
                await interaction.followup.send("No documents found in the knowledge base.")
                return
                
            # Get document UUIDs, original names, and metadata
            doc_items = []
            # The internal list document is also keyed by UUID now.
            # We need to find its UUID to skip it, or check its original_name.
            internal_list_doc_original_name = bot.document_manager._internal_list_doc_name

            for doc_uuid, meta in bot.document_manager.metadata.items():
                original_name = meta.get('original_name', doc_uuid) # Use UUID as fallback if name missing
                
                # Skip the internal list document itself
                if original_name == internal_list_doc_original_name:
                    continue
                
                chunks = meta.get('chunk_count', 'N/A')
                added_raw = meta.get('added', 'Unknown')
                try:
                    added_dt = datetime.fromisoformat(added_raw)
                    added_formatted = added_dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    added_formatted = added_raw
                
                # Store as tuple for sorting: (original_name_lower, display_string)
                display_string = f"{original_name} - {doc_uuid} ({chunks} chunks, Added: {added_formatted})"
                doc_items.append((original_name.lower(), display_string))

            doc_items.sort(key=lambda item: item[0]) # Sort by original name

            sorted_doc_display_strings = [item[1] for item in doc_items]
            header = f"Available documents ({len(sorted_doc_display_strings)}):"
            
            if not sorted_doc_display_strings: # Should be caught by the initial check, but good for safety
                await interaction.followup.send("No documents found in the knowledge base (excluding internal list).")
                return

            full_message_content = header + "\n" + "\n".join(sorted_doc_display_strings)
            
            # Split the entire message into chunks that will fit in a code block
            # Max length for content inside code block: 1990 (2000 - ```\n - \n```)
            message_chunks = split_message(full_message_content, max_length=1980)

            for i, chunk in enumerate(message_chunks):
                # The first chunk already contains the main header.
                # Subsequent chunks can get a "continued" header if we want, but the split is now safe.
                if i > 0:
                    # Prepend a continuation header to subsequent chunks
                    chunk = "Documents (continued):\n" + chunk
                
                # Wrap each chunk in a code block for sending
                formatted_chunk = f"```\n{chunk}\n```"
                await interaction.followup.send(formatted_chunk)
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            await interaction.followup.send(f"Error listing documents: {str(e)}")

    @bot.tree.command(name="remove_doc", description="Remove a document from the knowledge base (admin only)")
    @app_commands.describe(doc_uuid="UUID of the document to remove") # Changed parameter name
    @app_commands.check(check_permissions)
    async def remove_document(interaction: discord.Interaction, doc_uuid: str): # Changed parameter name
        await interaction.response.defer()
        try:
            if not doc_uuid:
                await interaction.followup.send("*neural error detected!* Please provide the document UUID.")
                return
            
            # Validate if it's a UUID (basic check)
            try:
                uuid.UUID(doc_uuid)
            except ValueError:
                await interaction.followup.send(f"*neural error detected!* '{doc_uuid}' is not a valid UUID format.")
                return
                
            # Get original name for user feedback before deletion, if possible
            original_name = bot.document_manager._get_original_name(doc_uuid) # Use the manager's method
            if original_name == doc_uuid and doc_uuid not in bot.document_manager.metadata: # If name is UUID and not in metadata
                 original_name = "Unknown (UUID not found in metadata)"


            success = await bot.document_manager.delete_document(doc_uuid) 
            if success:
                feedback_message = f"Document '{original_name}' (UUID: `{doc_uuid}`) removed successfully."
                # Check if this UUID was a tracked Google Doc and inform user
                tracked_gdocs_file = Path(bot.document_manager.base_dir) / "tracked_google_docs.json"
                if tracked_gdocs_file.exists():
                    try:
                        with open(tracked_gdocs_file, 'r', encoding='utf-8') as f:
                            tracked_docs = json.load(f)
                        if any(entry.get('internal_doc_uuid') == doc_uuid for entry in tracked_docs):
                            feedback_message += "\n*Note: This document was linked to a Google Doc. Its tracking entry has also been removed.*"
                        # No need to explicitly tell them to remove from tracking, as delete_document now handles it.
                    except Exception as e_gdoc:
                        logger.error(f"Error checking Google Doc tracking during remove_doc for UUID {doc_uuid}: {e_gdoc}")
                
                await interaction.followup.send(feedback_message)
            else:
                await interaction.followup.send(f"Document with UUID `{doc_uuid}` not found or could not be removed.")
        except Exception as e:
            logger.error(f"Error removing document via /remove_doc: {e}", exc_info=True)
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
            
            # Results tuple: (doc_uuid, original_name, chunk, similarity_score, image_id_if_applicable, chunk_index, total_chunks)
            for doc_uuid, original_name, chunk, similarity, image_id, chunk_index, total_chunks in results:
                # Format this result
                if image_id:
                    # This is an image search result
                    image_name = bot.image_manager.metadata.get(image_id, {}).get('name', "Unknown Image")
                    result_text = f"\n**IMAGE: {image_name}** (ID: {image_id}, similarity: {similarity:.2f}):\n"
                    result_text += f"```{bot.sanitize_discord_text(chunk[:300])}...```\n" # Assuming sanitize_discord_text exists
                else:
                    result_text = f"\n**From {original_name}** (UUID: `{doc_uuid}`, Chunk {chunk_index}/{total_chunks}) (similarity: {similarity:.2f}):\n"
                    result_text += f"```{bot.sanitize_discord_text(chunk[:300])}...```\n" # Assuming sanitize_discord_text exists
                
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

    @bot.tree.command(name="search_keyword", description="Search documents for a specific keyword")
    @app_commands.describe(keyword="Keyword to search for")
    async def search_keyword(interaction: discord.Interaction, keyword: str):
        """Return chunks containing the given keyword."""
        await interaction.response.defer()
        try:
            if not keyword:
                await interaction.followup.send("*neural error detected!* Please provide a keyword to search for.")
                return

            results = bot.document_manager.search_keyword(keyword, top_k=5)
            if not results:
                await interaction.followup.send("No occurrences found for that keyword.")
                return

            message = "Keyword results:\n"
            for doc_uuid, original_name, chunk, chunk_index, total_chunks in results:
                snippet = bot.sanitize_discord_text(chunk[:300])
                message += (
                    f"\n**From {original_name}** (UUID: `{doc_uuid}`, Chunk {chunk_index}/{total_chunks}):\n"
                    f"```{snippet}...```\n"
                )

            for chunk in split_message(message, max_length=1980):
                await interaction.followup.send(chunk)

        except Exception as e:
            await interaction.followup.send(f"Error searching for keyword: {str(e)}")

    @bot.tree.command(name="search_keyword_bm25", description="Search documents for a keyword using BM25")
    @app_commands.describe(keyword="Keyword to search for")
    async def search_keyword_bm25(interaction: discord.Interaction, keyword: str):
        """Return chunks containing the given keyword ranked by BM25."""
        await interaction.response.defer()
        try:
            if not keyword:
                await interaction.followup.send("*neural error detected!* Please provide a keyword to search for.")
                return

            results = bot.document_manager.search_keyword_bm25(keyword, top_k=5)
            if not results:
                await interaction.followup.send("No occurrences found for that keyword.")
                return

            message = "Keyword results (BM25):\n"
            for doc_uuid, original_name, chunk, chunk_index, total_chunks in results:
                snippet = bot.sanitize_discord_text(chunk[:300])
                message += (
                    f"\n**From {original_name}** (UUID: `{doc_uuid}`, Chunk {chunk_index}/{total_chunks}):\n"
                    f"```{snippet}...```\n"
                )

            for chunk in split_message(message, max_length=1980):
                await interaction.followup.send(chunk)

        except Exception as e:
            await interaction.followup.send(f"Error searching for keyword: {str(e)}")

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
            
            # Download the document content first
            await interaction.followup.send(f"*neural pathways connecting to Google Doc ID: {doc_id}... fetching content...*")
            
            name_for_doc_mgr = name or f"googledoc_{doc_id}" # Use custom name or default if name is None
            
            # The bot.refresh_single_google_doc method has been refactored to:
            # 1. Fetch content.
            # 2. Process it (including optional DOCX lore tagging).
            # 3. Add it to DocumentManager (which assigns/uses an internal_doc_uuid).
            # 4. Call DocumentManager.track_google_doc to update tracking.
            # 5. Return (success_bool, internal_doc_uuid_or_none).
            success_refresh, returned_uuid = await bot.refresh_single_google_doc(
                doc_id, 
                custom_name=name_for_doc_mgr, # Pass the determined name
                interaction_for_feedback=interaction # For followup messages during the process
            )

            if success_refresh and returned_uuid:
                await interaction.followup.send(f"Google Doc '{name_for_doc_mgr}' (ID: {doc_id}) processed and added to knowledge base with internal UUID: `{returned_uuid}`. Tracking established.")
            elif success_refresh and not returned_uuid: 
                 await interaction.followup.send(f"Google Doc '{name_for_doc_mgr}' (ID: {doc_id}) content may have been fetched, but it failed to be added to the document manager or a UUID was not returned.")
            else:
                await interaction.followup.send(f"*neural connection established but document download or processing failed for '{name_for_doc_mgr}' (ID: {doc_id}). Try refreshing later.*")
        except Exception as e:
            logger.error(f"Error adding Google Doc: {e}", exc_info=True)
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
            for doc_entry in tracked_docs: # doc_entry is now like {"google_doc_id": ..., "internal_doc_uuid": ..., "original_name_at_import": ...}
                google_doc_id = doc_entry.get('google_doc_id')
                internal_uuid = doc_entry.get('internal_doc_uuid')
                original_name = doc_entry.get('original_name_at_import', f"googledoc_{google_doc_id}")
                
                doc_url = f"<https://docs.google.com/document/d/{google_doc_id}>"
                display_string = f"**{original_name}**\n  Google Doc ID: `{google_doc_id}`\n  Internal UUID: `{internal_uuid}`\n  URL: {doc_url}"
                doc_list_items.append((original_name.lower(), display_string))

            doc_list_items.sort(key=lambda item: item[0])

            response_header = f"*Accessing neural connections to Google Docs...*\n\n**TRACKED GOOGLE DOCUMENTS ({len(doc_list_items)}):**\n"
            
            if not doc_list_items: # Should be caught earlier, but good for safety
                 await interaction.followup.send("No Google Docs are currently tracked.")
                 return

            full_response_content = "\n".join([item[1] for item in doc_list_items])
            
            message_chunks = split_message(full_response_content, max_length=1900) # Leave space for header

            for i, chunk_content in enumerate(message_chunks):
                current_header = response_header if i == 0 else "Tracked Google Documents (continued):\n"
                await interaction.followup.send(f"{current_header}\n{chunk_content}")
        except Exception as e:
            logger.error(f"Error listing Google Docs: {e}", exc_info=True)
            await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to list Google Docs.")

    @bot.tree.command(name="rename_document", description="Rename any document, Google Doc, or lorebook (admin only)")
    @app_commands.describe(
        current_name="Document UUID or current original name",
        new_name="New original name for the document"
    )
    @app_commands.check(check_permissions)
    async def rename_document(interaction: discord.Interaction, current_name: str, new_name: str): # Parameter name 'current_name' kept for user input flexibility
        await interaction.response.defer()
        try:
            if not current_name or not new_name:
                await interaction.followup.send("*neural error detected!* Both current identifier (UUID or name) and new name are required.")
                return

            # Attempt to use current_name as UUID first
            doc_uuid_to_rename = current_name
            try:
                uuid.UUID(doc_uuid_to_rename) # Validate if it's a UUID
                if doc_uuid_to_rename not in bot.document_manager.metadata:
                    # If it looked like a UUID but wasn't found, try finding by original name
                    found_uuid = None
                    for d_uuid, meta in bot.document_manager.metadata.items():
                        if meta.get('original_name') == current_name:
                            found_uuid = d_uuid
                            break
                    if not found_uuid:
                        await interaction.followup.send(f"Document with UUID or original name '{current_name}' not found.")
                        return
                    doc_uuid_to_rename = found_uuid
            except ValueError: # Not a UUID, so assume it's an original name
                found_uuid = None
                for d_uuid, meta in bot.document_manager.metadata.items():
                    if meta.get('original_name') == current_name:
                        found_uuid = d_uuid
                        break
                if not found_uuid:
                    await interaction.followup.send(f"Document with original name '{current_name}' not found.")
                    return
                doc_uuid_to_rename = found_uuid
                
            result = await bot.document_manager.rename_document(doc_uuid_to_rename, new_name)
            await interaction.followup.send(f"*Synaptic pathways reconfiguring...*\n{result}")
        except Exception as e:
            logger.error(f"Error renaming document: {e}", exc_info=True)
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
            
            # Try to find and remove the doc from tracking
            google_doc_id_to_remove = None
            internal_uuid_of_gdoc_content = None
            original_name_of_gdoc = None

            for i, doc_entry in enumerate(tracked_docs):
                gdoc_id = doc_entry.get('google_doc_id')
                custom_name = doc_entry.get('original_name_at_import') # Use the consistent name field
                
                if gdoc_id == identifier or \
                   (custom_name and custom_name == identifier) or \
                   (extracted_id and gdoc_id == extracted_id):
                    
                    google_doc_id_to_remove = gdoc_id
                    internal_uuid_of_gdoc_content = doc_entry.get('internal_doc_uuid')
                    original_name_of_gdoc = custom_name or f"googledoc_{gdoc_id}"
                    tracked_docs.pop(i) # Remove from list
                    break 
            
            if not google_doc_id_to_remove:
                await interaction.followup.send(f"*Hmm, I couldn't find a Google Doc matching '{identifier}' in my tracking list.*")
                return
            
            # Save updated tracking list (with the entry removed)
            with open(tracked_file, 'w', encoding='utf-8') as f:
                json.dump(tracked_docs, f, indent=2)
            
            feedback_message = f"Google Doc '{original_name_of_gdoc}' (ID: {google_doc_id_to_remove}) removed from tracking."

            # Now, delete the local copy using its internal_doc_uuid if it exists
            if internal_uuid_of_gdoc_content:
                logger.info(f"Attempting to delete local document content for Google Doc ID {google_doc_id_to_remove} using internal UUID: {internal_uuid_of_gdoc_content}")
                delete_success = await bot.document_manager.delete_document(internal_uuid_of_gdoc_content)
                if delete_success:
                    feedback_message += f"\nAssociated local document (UUID: `{internal_uuid_of_gdoc_content}`) also removed from knowledge base."
                else:
                    feedback_message += f"\nNote: Could not remove associated local document (UUID: `{internal_uuid_of_gdoc_content}`). It might have already been deleted or was not found."
            else:
                feedback_message += "\nNo internal UUID was associated with this Google Doc for local file deletion."

            await interaction.followup.send(feedback_message)
                
        except Exception as e:
            logger.error(f"Error removing Google Doc: {e}", exc_info=True)
            await interaction.followup.send(f"*my enhanced brain experienced an error!* couldn't remove document: {str(e)}")
    
    @bot.tree.command(name="retrieve_file", description="Retrieve a document by its UUID")
    @app_commands.describe(
        doc_uuid="UUID of the document to retrieve"
    )
    async def retrieve_file(interaction: discord.Interaction, doc_uuid: str):
        await interaction.response.defer()
        try:
            if not doc_uuid:
                await interaction.followup.send("*neural error detected!* Please provide the document UUID.")
                return
            
            try:
                uuid.UUID(doc_uuid) # Validate format
            except ValueError:
                await interaction.followup.send(f"*neural error detected!* '{doc_uuid}' is not a valid UUID format.")
                return

            doc_mgr = bot.document_manager
            if doc_uuid not in doc_mgr.metadata:
                await interaction.followup.send(f"*neural error detected!* Document with UUID `{doc_uuid}` not found.")
                return

            file_path = doc_mgr.base_dir / f"{doc_uuid}.txt"
            if not file_path.exists() or not file_path.is_file():
                # This case should be rare if metadata exists, but good to handle.
                # Try to reconstruct from chunks if file is missing.
                if doc_uuid in doc_mgr.chunks:
                    logger.warning(f"File {file_path} not found for UUID {doc_uuid}. Reconstructing from chunks for retrieval.")
                    content_to_send = "\n\n".join(doc_mgr.chunks[doc_uuid])
                    original_name = doc_mgr.metadata[doc_uuid].get('original_name', f"{doc_uuid}.txt")
                    # Send as a text file from string
                    temp_file_path = Path(f"./temp_files/{original_name}") # Ensure temp_files dir exists
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_file_path.write_text(content_to_send, encoding='utf-8')
                    discord_file = discord.File(temp_file_path, filename=original_name)
                    await interaction.followup.send(
                        content=f"*Neural pathways accessing storage...* Here is the document (reconstructed): `{original_name}` (UUID: `{doc_uuid}`)",
                        file=discord_file
                    )
                    temp_file_path.unlink() # Clean up temp file
                    return
                else:
                    await interaction.followup.send(f"*neural error detected!* Document file for UUID `{doc_uuid}` not found and no chunks available.")
                    return
            
            original_name = doc_mgr.metadata[doc_uuid].get('original_name', f"{doc_uuid}.txt")
            discord_file_to_send = discord.File(file_path, filename=original_name) # Send with user-friendly original_name
            
            await interaction.followup.send(
                content=f"*Neural pathways accessing storage...* Here is the document: `{original_name}` (UUID: `{doc_uuid}`)",
                file=discord_file_to_send
            )
        
        except Exception as e:
            logger.error(f"Error retrieving file by UUID '{doc_uuid}': {e}", exc_info=True)
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

    @bot.tree.command(name="force_refresh_googledocs", description="Force refresh/process all tracked Google Docs (admin only)")
    @app_commands.check(check_permissions)
    async def force_refresh_googledocs(interaction: discord.Interaction):
        """Manually trigger the refresh and processing for ALL tracked Google Docs, bypassing change detection."""
        await interaction.response.defer()
        try:
            await interaction.followup.send("*neural pathways activating... initiating FORCE refresh and processing sequence for ALL Google Docs...*")
            
            # Call the refresh method with force_process=True
            await bot.refresh_google_docs(force_process=True)
            
            await interaction.followup.send("*neural synchronization complete!* Google Docs force refresh finished.")
        except Exception as e:
            logger.error(f"Error during manual Google Docs force refresh: {e}")
            await interaction.followup.send(f"*neural circuit overload!* An error occurred during force refresh: {str(e)}")

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
                nonlocal last_update_time
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
                    # message_count_so_far = message_count # Unused
                    # attachment_count_so_far = attachment_count # Unused
                    # reaction_count_so_far = reaction_count # Unused
                    # embed_count_so_far = embed_count # Unused
                    
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
            
            # Save documents using the DocumentManager
            # DocumentManager will handle UUID assignment and file storage.
            # We still need to create temporary files for Discord upload if the archive is split.
            
            processed_doc_info = [] # To store (original_name, uuid, temp_file_path_for_discord)

            temp_archive_dir = Path("./temp_files/archive")
            temp_archive_dir.mkdir(parents=True, exist_ok=True)

            for original_doc_name, doc_content_to_save in documents:
                # Add to document manager, which returns the UUID
                # The original_doc_name here will be like "channel_archive_general_20230101_1200_part1.txt"
                doc_uuid_assigned = await bot.document_manager.add_document(original_name=original_doc_name, content=doc_content_to_save)
                
                if not doc_uuid_assigned:
                    logger.error(f"Failed to add archived document part '{original_doc_name}' to DocumentManager.")
                    await interaction.followup.send(f"Error: Failed to save document part '{original_doc_name}' to the knowledge base.")
                    continue # Skip this part if adding failed

                # For sending to Discord, we might need to save it temporarily if it's large or for consistency
                safe_temp_filename = sanitize_filename(original_doc_name) # Sanitize for local temp file path
                temp_discord_upload_path = temp_archive_dir / safe_temp_filename
                try:
                    temp_discord_upload_path.write_text(doc_content_to_save, encoding='utf-8')
                    processed_doc_info.append({
                        "original_name": original_doc_name, # Keep original for Discord display name
                        "uuid": doc_uuid_assigned,
                        "discord_path": temp_discord_upload_path,
                        "safe_filename_for_discord": safe_temp_filename # Store the sanitized name if needed for discord.File
                    })
                except Exception as e_write_temp:
                    logger.error(f"Failed to write temporary archive part {original_doc_name} (safe: {safe_temp_filename}) for Discord upload: {e_write_temp}")
                    processed_doc_info.append({
                        "original_name": original_doc_name,
                        "uuid": doc_uuid_assigned,
                        "discord_path": None, 
                        "safe_filename_for_discord": safe_temp_filename
                    })

            if not processed_doc_info:
                await interaction.followup.send("Failed to process and save any archive parts.")
                return

            # Create downloadable files
            if len(processed_doc_info) == 1:
                # Single file case
                doc_info = processed_doc_info[0]
                discord_file_to_send = None
                if doc_info["discord_path"] and doc_info["discord_path"].exists():
                    # Use original_name for the filename presented to the user in Discord.
                    discord_file_to_send = discord.File(doc_info["discord_path"], filename=doc_info["original_name"])
                
                feedback_content = (
                    f"*Neural archiving complete!* Successfully archived {len(messages)} messages from #{channel.name}.\n\n"
                    f"• Document Name: `{doc_info['original_name']}` (UUID: `{doc_info['uuid']}`)\n" # Display original name
                    f"• Messages: {message_count}\n"
                    f"• Attachments: {attachment_count}\n"
                    f"• Reactions: {reaction_count}\n"
                    f"• Embeds: {embed_count}\n\n"
                    f"The archive has been added to my knowledge base."
                )
                if discord_file_to_send:
                    feedback_content += " It is also available as a downloadable file below:"
                    await interaction.followup.send(content=feedback_content, file=discord_file_to_send)
                else:
                    feedback_content += " (Could not prepare file for download)."
                    await interaction.followup.send(content=feedback_content)

            else: # Multiple parts
                first_doc_info = processed_doc_info[0]
                discord_file_to_send = None
                if first_doc_info["discord_path"] and first_doc_info["discord_path"].exists():
                     # Use original_name for the filename presented to the user in Discord.
                     discord_file_to_send = discord.File(first_doc_info["discord_path"], filename=first_doc_info["original_name"])

                doc_names_and_uuids = ', '.join([f"`{info['original_name']}` (UUID: `{info['uuid']}`)" for info in processed_doc_info]) # Display original names
                
                feedback_content = (
                    f"*Neural archiving complete!* Successfully archived {len(messages)} messages from #{channel.name}.\n\n"
                    f"• Document parts: {len(processed_doc_info)}\n"
                    f"• Document Names & UUIDs: {doc_names_and_uuids}\n" # Display original names
                    f"• Total Messages: {message_count}\n"
                    f"• Total Attachments: {attachment_count}\n"
                    f"• Total Reactions: {reaction_count}\n"
                    f"• Total Embeds: {embed_count}\n\n"
                    f"The archive parts have been added to my knowledge base."
                )
                if discord_file_to_send:
                    feedback_content += " The first part is attached below. Use `/retrieve_file <UUID>` to download other parts."
                    await interaction.followup.send(content=feedback_content, file=discord_file_to_send)
                else:
                    feedback_content += " (Could not prepare first part for download). Use `/retrieve_file <UUID>` to download parts."
                    await interaction.followup.send(content=feedback_content)

            # Clean up temporary archive files
            for info in processed_doc_info:
                if info["discord_path"] and info["discord_path"].exists():
                    try: info["discord_path"].unlink()
                    except Exception as e_unlink: logger.error(f"Failed to delete temp archive file {info['discord_path']}: {e_unlink}")
            
        except Exception as e:
            logger.error(f"Error archiving channel: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            await interaction.followup.send(f"*neural circuit overload!* An error occurred while archiving the channel: {str(e)}")

    @bot.tree.command(name="summarize_doc", description="Generate a summary of a document by its UUID or original name")
    @app_commands.describe(identifier="UUID or original name of the document to summarize")
    async def summarize_document(interaction: discord.Interaction, identifier: str):
        """Generates a summary of the specified document using an LLM."""
        await interaction.response.defer()
        try:
            if not identifier:
                await interaction.followup.send("*neural error detected!* Please provide a document UUID or original name.")
                return

            doc_mgr = bot.document_manager
            doc_uuid_to_summarize = None
            original_name_for_display = identifier # Default to identifier

            # Try to resolve identifier to UUID
            try:
                uuid.UUID(identifier) # Check if it's a UUID format
                if identifier in doc_mgr.metadata:
                    doc_uuid_to_summarize = identifier
                    original_name_for_display = doc_mgr.metadata[identifier].get('original_name', identifier)
            except ValueError: # Not a UUID, assume it's an original name
                pass

            if not doc_uuid_to_summarize: # If not resolved as UUID or not found by UUID, search by original name
                for d_uuid, meta in doc_mgr.metadata.items():
                    if meta.get('original_name') == identifier:
                        doc_uuid_to_summarize = d_uuid
                        original_name_for_display = identifier
                        break
            
            if not doc_uuid_to_summarize:
                await interaction.followup.send(f"*neural error!* Document with identifier '{identifier}' not found.")
                return

            # --- Fetch document content using UUID ---
            doc_content = None
            doc_file_path = doc_mgr.base_dir / f"{doc_uuid_to_summarize}.txt"
            if doc_file_path.exists() and doc_file_path.is_file():
                try:
                    doc_content = doc_file_path.read_text(encoding='utf-8-sig')
                except Exception as read_err:
                    logger.error(f"Error reading document file {doc_file_path} for summarization: {read_err}")
                    await interaction.followup.send(f"*neural error!* Failed to read the document file: {read_err}")
                    return
            elif doc_uuid_to_summarize in doc_mgr.chunks: # Fallback to chunks
                logger.warning(f"Document file for UUID {doc_uuid_to_summarize} not found. Reconstructing from chunks for summarization.")
                doc_content = "\n\n".join(doc_mgr.chunks[doc_uuid_to_summarize])
            
            if not doc_content or not doc_content.strip():
                await interaction.followup.send(f"*neural analysis complete!* The document '{original_name_for_display}' (UUID: `{doc_uuid_to_summarize}`) appears to be empty.")
                return

            # --- Prepare for LLM call ---
            status_message = await interaction.followup.send(f"*neural pathways activating... preparing summary for '{original_name_for_display}' (UUID: `{doc_uuid_to_summarize}`)...*")

            # Use user's preferred model or default
            preferred_model = bot.user_preferences_manager.get_preferred_model(
                str(interaction.user.id),
                default_model=bot.config.LLM_MODEL
            )
            # Select a capable model (prefer larger context if possible, but use preferred)
            # For summarization, a model with good comprehension is needed.
            summary_model = preferred_model # Use user's preference for consistency

            # Construct the prompt
            summary_system_prompt = "You are an AI assistant tasked with summarizing documents accurately and concisely. Respond only with the summary."
            prompt_messages = [
                {"role": "system", "content": summary_system_prompt},
                {"role": "user", "content": f"Please summarize the following document titled '{original_name_for_display}':\n\n<document_content>\n{doc_content}\n</document_content>\n\nProvide a concise summary of the key information."}
            ]

            await status_message.edit(content=f"*neural core processing... generating summary for '{original_name_for_display}' (UUID: `{doc_uuid_to_summarize}`) using {summary_model}...*")

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
                    response_header = f"**Summary for Document: {original_name_for_display}** (UUID: `{doc_uuid_to_summarize}`, Generated by {actual_model})\n\n"
                    
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
                await status_message.edit(content=f"*synaptic failure detected!* I apologize, but I couldn't generate a summary for '{original_name_for_display}' (UUID: `{doc_uuid_to_summarize}`) right now.")

        except Exception as e:
            logger.error(f"Error summarizing document '{identifier}': {e}", exc_info=True)
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

    @bot.tree.command(name="view_chunk", description="View the content of a specific document chunk by UUID or name")
    @app_commands.describe(
        identifier="UUID or original name of the document",
        chunk_index="The index of the chunk to view (starting from 1)",
        contextualized="View the contextualized version used for embeddings (default: False)"
    )
    async def view_chunk(interaction: discord.Interaction, identifier: str, chunk_index: int, contextualized: bool = False):
        """Displays the content of a specific chunk from a document."""
        await interaction.response.defer()
        try:
            if not identifier:
                await interaction.followup.send("*neural error detected!* Please provide a document UUID or original name.")
                return
            if chunk_index <= 0:
                await interaction.followup.send("*neural error detected!* Chunk index must be 1 or greater.")
                return

            doc_mgr = bot.document_manager
            doc_uuid_to_view = None
            original_name_for_display = identifier # Default

            # Try to resolve identifier to UUID
            try:
                uuid.UUID(identifier) # Check if it's a UUID format
                if identifier in doc_mgr.metadata:
                    doc_uuid_to_view = identifier
                    original_name_for_display = doc_mgr.metadata[identifier].get('original_name', identifier)
            except ValueError: # Not a UUID, assume it's an original name
                pass

            if not doc_uuid_to_view: # If not resolved as UUID or not found by UUID, search by original name
                for d_uuid, meta in doc_mgr.metadata.items():
                    if meta.get('original_name') == identifier:
                        doc_uuid_to_view = d_uuid
                        original_name_for_display = identifier
                        break
            
            if not doc_uuid_to_view:
                await interaction.followup.send(f"*neural error!* Document with identifier '{identifier}' not found.")
                return

            # Check if contextualised chunks are globally disabled
            use_contextualised = bot.config.USE_CONTEXTUALISED_CHUNKS if hasattr(bot, 'config') and hasattr(bot.config, 'USE_CONTEXTUALISED_CHUNKS') else True
            
            # Determine which chunk list to use
            if contextualized and not use_contextualised:
                await interaction.followup.send("⚠️ **Warning:** Contextualised chunks are currently disabled in the configuration. Showing original chunks instead.")
                chunk_source_dict = doc_mgr.chunks
                chunk_type_str = "Original (contextualised disabled)"
            else:
                chunk_source_dict = doc_mgr.contextualized_chunks if contextualized else doc_mgr.chunks
                chunk_type_str = "Contextualized" if contextualized else "Original"
            
            doc_chunks_list = chunk_source_dict.get(doc_uuid_to_view)
            total_doc_chunks = len(doc_chunks_list) if doc_chunks_list is not None else 0

            if contextualized and not doc_chunks_list and doc_uuid_to_view in doc_mgr.chunks: # Fallback for contextualized
                logger.warning(f"Contextualized chunk requested for '{original_name_for_display}' (UUID: {doc_uuid_to_view}), but not found. Falling back to original.")
                doc_chunks_list = doc_mgr.chunks.get(doc_uuid_to_view)
                total_doc_chunks = len(doc_chunks_list) if doc_chunks_list is not None else 0
                chunk_type_str = "Original (Fallback)"

            if doc_chunks_list is None: # Should not happen if UUID was found in metadata, implies missing chunks
                await interaction.followup.send(f"*neural error!* Chunk data not found for document '{original_name_for_display}' (UUID: `{doc_uuid_to_view}`).")
                return

            zero_based_idx = chunk_index - 1
            if not (0 <= zero_based_idx < total_doc_chunks):
                await interaction.followup.send(f"*neural error!* Invalid chunk index {chunk_index}. Document '{original_name_for_display}' (UUID: `{doc_uuid_to_view}`) has {total_doc_chunks} chunks.")
                return

            chunk_content_to_display = doc_chunks_list[zero_based_idx]
            response_header = f"**{chunk_type_str} Chunk {chunk_index}/{total_doc_chunks} from Document: {original_name_for_display}** (UUID: `{doc_uuid_to_view}`)"

            # Sanitize the raw chunk content *before* splitting
            # Assuming bot has sanitize_discord_text method, otherwise remove/adjust
            sanitized_content = chunk_content_to_display # Default if sanitize method doesn't exist or causes issues
            if hasattr(bot, 'sanitize_discord_text') and callable(bot.sanitize_discord_text):
                 sanitized_content = bot.sanitize_discord_text(chunk_content_to_display)
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
            logger.error(f"Error viewing chunk {chunk_index} from '{original_name_for_display}': {e}")
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

    @bot.tree.command(name="ressummarize_docs", description="Force regenerate summaries for all documents (admin only)")
    @app_commands.check(check_permissions)
    async def ressummarize_docs(interaction: discord.Interaction):
        """Forces the regeneration of all document summaries."""
        await interaction.response.defer()
        try:
            await interaction.followup.send("*neural core activating... initiating full summary regeneration for all documents. This may take some time...*")
            
            updated_count = await bot.document_manager.regenerate_all_summaries()
            
            await interaction.followup.send(f"*neural processing complete!* Successfully regenerated and updated summaries for **{updated_count}** documents.")
            
        except Exception as e:
            logger.error(f"Error during summary regeneration command: {e}", exc_info=True)
            await interaction.followup.send(f"*neural circuit overload!* An error occurred during summary regeneration: {str(e)}")

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

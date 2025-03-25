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
from utils.helpers import split_message

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all document management commands with the bot."""
    
    @bot.tree.command(name="add_info", description="Add new text to Publicia's mind for retrieval")
    @app_commands.describe(
        name="Name of the document",
        content="Content of the document"
    )
    async def add_document(interaction: discord.Interaction, name: str, content: str):
        await interaction.response.defer()
        try:
            if not name or not content:
                await interaction.followup.send("*neural error detected!* Both name and content are required.")
                return
                
            lorebooks_path = bot.document_manager.get_lorebooks_path()
            txt_path = lorebooks_path / f"{name}.txt"
            txt_path.write_text(content, encoding='utf-8')
            
            bot.document_manager.add_document(name, content)
            await interaction.followup.send(f"Added document: {name}\nSaved to: {txt_path}")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            await interaction.followup.send(f"Error adding document: {str(e)}")
    
    @bot.command(name="add_doc", brief="Add a new document to the knowledge base. Usage: Publicia! add_doc \"Document Name\"")
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
                
            # Get all documents first
            doc_items = []
            for doc_name, meta in bot.document_manager.metadata.items():
                chunks = meta['chunk_count']
                added = meta['added']
                doc_items.append(f"{doc_name} - {chunks} chunks (Added: {added})")
            
            # Create header
            header = "Available documents:"
            
            # Split into chunks, allowing room for code block formatting
            doc_chunks = split_message("\n".join(doc_items), max_length=1900)  # Leave room for formatting
            
            for i, chunk in enumerate(doc_chunks):
                # Format each chunk as a separate code block
                formatted_chunk = f"{header if i == 0 else 'Documents (continued):'}\n```\n{chunk}\n```"
                await interaction.followup.send(formatted_chunk)
                
        except Exception as e:
            await interaction.followup.send(f"Error listing documents: {str(e)}")

    @bot.tree.command(name="remove_doc", description="Remove a document from the knowledge base")
    @app_commands.describe(name="Name of the document to remove")
    async def remove_document(interaction: discord.Interaction, name: str):
        await interaction.response.defer()
        try:
            if not name:
                await interaction.followup.send("*neural error detected!* Please provide a document name.")
                return
                
            success = bot.document_manager.delete_document(name)
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
                
            results = bot.document_manager.search(query, top_k=5)
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

    @bot.tree.command(name="add_googledoc", description="Add a Google Doc to the tracked list")
    @app_commands.describe(
        doc_url="Google Doc URL or ID",
        name="Custom name for the document (optional)"
    )
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
                
            response = "*accessing neural connections to google docs...*\n\n**TRACKED DOCUMENTS**\n"
            for doc in tracked_docs:
                doc_id = doc['id']
                name = doc.get('custom_name') or f"googledoc_{doc_id}.txt"
                doc_url = f"<https://docs.google.com/document/d/{doc_id}>"
                response += f"\n{name} - URL: {doc_url}"
            
            # Split the message to avoid Discord's 2000 character limit
            for chunk in split_message(response):
                await interaction.followup.send(chunk)
        except Exception as e:
            logger.error(f"Error listing Google Docs: {e}")
            await interaction.followup.send("*neural circuit overload!* I encountered an error while trying to list Google Docs.")

    @bot.tree.command(name="rename_document", description="Rename any document, Google Doc, or lorebook")
    @app_commands.describe(
        current_name="Current name of the document to rename",
        new_name="New name for the document"
    )
    async def rename_document(interaction: discord.Interaction, current_name: str, new_name: str):
        await interaction.response.defer()
        try:
            if not current_name or not new_name:
                await interaction.followup.send("*neural error detected!* Both current name and new name are required.")
                return
                
            result = bot.document_manager.rename_document(current_name, new_name)
            await interaction.followup.send(f"*synaptic pathways reconfiguring...*\n{result}")
        except Exception as e:
            logger.error(f"Error renaming document: {e}")
            await interaction.followup.send(f"*neural pathway error!* couldn't rename document: {str(e)}")

    @bot.tree.command(name="remove_googledoc", description="Remove a Google Doc from the tracked list")
    @app_commands.describe(
        identifier="Google Doc ID, URL, or custom name to remove"
    )
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
                
            # Remove local file if it exists
            local_file_path = Path(bot.document_manager.base_dir) / local_file_name
            file_removed = False
            
            if local_file_path.exists():
                try:
                    success = bot.document_manager.delete_document(local_file_name)
                    if success:
                        file_removed = True
                    else:
                        await interaction.followup.send(f"Document tracked, but file not found in document manager: {local_file_name}")
                except Exception as e:
                    await interaction.followup.send(f"Error removing document: {str(e)}")
            
            response = f"*I've surgically removed the neural connection to {doc_name}*\n*url: {doc_url}*"
            if file_removed:
                response += f"\n*and removed the local document file ({local_file_name})*"
            
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

    @bot.tree.command(name="archive_channel", description="Archive messages from a Discord channel as a document")
    @app_commands.describe(
        channel="The channel to archive messages from",
        limit="Maximum number of messages to archive (leave empty for no limit)",
        document_name="Custom name for the archived document (optional)"
    )
    async def archive_channel(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = None, document_name: str = None):
        await interaction.response.defer()
        try:
            # Check if user has admin permissions
            if not interaction.user.guild_permissions.administrator:
                await interaction.followup.send("*neural access denied!* This command requires administrator permissions.")
                return
                
            # Send initial status message
            status_message = await interaction.followup.send("*neural archiving process initiated...* scanning channel messages")
            
            # Determine document name if not provided
            if not document_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                document_name = f"channel_archive_{channel.name}_{timestamp}"
            
            # Ensure document name has .txt extension
            if not document_name.endswith('.txt'):
                document_name += '.txt'
                
            # Initialize content with header
            content = f"# DISCORD CHANNEL ARCHIVE: {channel.name}\n"
            content += f"# Server: {interaction.guild.name}\n"
            content += f"# Archived by: {interaction.user.name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"# Message count: {'All' if limit is None else limit}\n\n"
            
            # Initialize counters
            message_count = 0
            attachment_count = 0
            reaction_count = 0
            embed_count = 0
            
            # Update status message with progress info
            await status_message.edit(content="*neural pathways connecting to channel history...* retrieving messages")
            
            # Fetch messages
            messages = []
            last_update_time = time.time()
            update_interval = 2.0  # Update status every 2 seconds
            
            async for message in channel.history(limit=limit):
                messages.append(message)
                message_count += 1
                
                # Show progress updates for large archives
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    await status_message.edit(content=f"*neural archiving in progress...* retrieved {message_count} messages so far")
                    last_update_time = current_time
            
            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda m: m.created_at)
            
            # Update status
            await status_message.edit(content=f"*neural formatting process initiated...* formatting {len(messages)} messages")
            
            # Process messages
            for i, message in enumerate(messages):
                # Show progress for large archives
                if i % 500 == 0 and i > 0:
                    await status_message.edit(content=f"*neural formatting in progress...* processed {i}/{len(messages)} messages")
                
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
                content += f"\n[{timestamp}] {author_name}{author_roles}:\n"
                
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
                        content += f"    {line}\n"
                
                # Add attachments
                if message.attachments:
                    content += "    [Attachments]\n"
                    for attachment in message.attachments:
                        content += f"    • {attachment.filename} - {attachment.url}\n"
                        attachment_count += 1
                
                # Add embeds
                if message.embeds:
                    content += "    [Embeds]\n"
                    for embed in message.embeds:
                        embed_count += 1
                        if embed.title:
                            content += f"    • Title: {embed.title}\n"
                        if embed.description:
                            content += f"      Description: {embed.description[:100]}{'...' if len(embed.description) > 100 else ''}\n"
                
                # Add reactions
                if message.reactions:
                    reaction_str = "    [Reactions] "
                    for reaction in message.reactions:
                        reaction_count += 1
                        emoji = reaction.emoji if isinstance(reaction.emoji, str) else reaction.emoji.name
                        reaction_str += f"{emoji} ({reaction.count}) "
                    content += reaction_str + "\n"
            
            # Add footer with statistics
            content += f"\n\n# ARCHIVE SUMMARY\n"
            content += f"# Total messages: {message_count}\n"
            content += f"# Attachments: {attachment_count}\n"
            content += f"# Reactions: {reaction_count}\n"
            content += f"# Embeds: {embed_count}\n"
            
            # Update status
            await status_message.edit(content=f"*neural storage process initiated...* saving {len(messages)} messages to document system")
            
            # Save as a document file
            lorebooks_path = bot.document_manager.get_lorebooks_path()
            txt_path = lorebooks_path / document_name
            txt_path.write_text(content, encoding='utf-8')
            
            # Add to document manager
            await bot.document_manager.add_document(document_name, content)
            
            # Create downloadable file
            file = discord.File(txt_path, filename=document_name)
            
            # Send completion message with file
            await interaction.followup.send(
                content=f"*neural archiving complete!* Successfully archived {message_count} messages from #{channel.name}.\n\n"
                        f"• Document name: `{document_name}`\n"
                        f"• Attachments: {attachment_count}\n"
                        f"• Reactions: {reaction_count}\n"
                        f"• Embeds: {embed_count}\n\n"
                        f"The archive has been added to my knowledge base and is also available as a downloadable file below:",
                file=file
            )
            
        except Exception as e:
            logger.error(f"Error archiving channel: {e}")
            await interaction.followup.send(f"*neural circuit overload!* An error occurred while archiving the channel: {str(e)}")

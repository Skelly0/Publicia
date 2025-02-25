# Publicia Bot Documentation

*your enhanced neural interface to ledus banum 77 and imperial lore!*

## table of contents
- [overview](#overview)
- [setup & installation](#setup--installation)
- [configuration](#configuration)
- [features](#features)
- [commands](#commands)
- [architecture](#architecture)
- [troubleshooting](#troubleshooting)
- [developer reference](#developer-reference)

---

## overview

publicia is a discord bot that serves as an intelligent neural interface to your lore documents. it uses advanced embedding technology to understand and respond to questions about your setting, maintaining personalized conversation history with each user.

### key capabilities
- answers questions about ledus banum 77 and imperial lore
- remembers conversations with individual users
- imports documents from text files or google docs
- uses AI to provide context-aware responses
- semantic search across all knowledge documents

---

## setup & installation

### prerequisites
- python 3.8 or higher
- discord bot token (from discord developer portal)
- openrouter API key
- access to a language model (via openrouter)

### installation steps

1. clone the repository or download the source code

2. install required packages:
```bash
pip install discord.py python-dotenv sentence-transformers numpy aiohttp apscheduler
```

3. create a `.env` file in the root directory with the following:
```
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=your_preferred_model
```

4. make sure you have `system_prompt.py` in the same directory that contains the SYSTEM_PROMPT variable

5. create directories for document storage:
```
documents/
lorebooks/
conversations/
```

6. run the bot:
```bash
python PubliciaV2.py
```

---

## configuration

publicia uses environment variables for configuration:

| variable | required | default | description |
|----------|----------|---------|-------------|
| DISCORD_BOT_TOKEN | yes | none | token for discord bot authentication |
| OPENROUTER_API_KEY | yes | none | API key for openrouter service |
| LLM_MODEL | yes | none | model identifier for AI responses |
| API_TIMEOUT | no | 30 | timeout in seconds for API calls |
| MAX_RETRIES | no | 3 | maximum retry attempts for API calls |

---

## features

### semantic document search
publicia uses embeddings to understand the meaning behind user questions, not just keywords! this allows for more natural queries and better answers.

### conversation memory
the bot remembers interactions with each user, providing continuity across conversations. each user has their own conversation history that provides context for the AI.

### google docs integration
automatically fetch and update content from google docs to keep your lore fresh without manual updates.

### ai-powered responses
combines conversation history, relevant document context, and advanced AI to give informative and contextually appropriate answers.

### admin controls
moderators can manage users and documents, including banning problematic users if needed.

---

## commands

### general commands
| command | description | usage |
|---------|-------------|-------|
| /query | ask publicia a question | `/query question:what is the capital of the empire?` |
| /listcommands | show all available commands | `/listcommands` |
| /lobotomise | clear your conversation history | `/lobotomise` |

### document management
| command | description | usage |
|---------|-------------|-------|
| /adddoc | add a document to the knowledge base | `/adddoc name:empire_history content:long ago...` |
| /listdocs | list all available documents | `/listdocs` |
| /removedoc | remove a document | `/removedoc name:outdated_lore` |
| /searchdocs | search documents directly | `/searchdocs query:emperor's throne` |

### google docs integration
| command | description | usage |
|---------|-------------|-------|
| /add_googledoc | track a google doc | `/add_googledoc doc_id:1abc... name:imperial_navy` |
| /list_googledocs | show tracked google docs | `/list_googledocs` |

### moderation (admin only)
| command | description | usage |
|---------|-------------|-------|
| /ban_user | ban user from using the bot | `/ban_user user:@troublemaker` |
| /unban_user | unban a previously banned user | `/unban_user user:@reformed_user` |

### special message functions
- mention the bot to ask questions directly
- include "LOBOTOMISE" in a message to clear your history

---

## architecture

publicia consists of four main components:

1. **documentmanager**: handles document storage and retrieval
   - uses sentence transformers for embedding generation
   - chunks documents for better context retrieval
   - manages document metadata and storage

2. **conversationmanager**: manages user interaction history
   - stores conversations as JSON
   - retrieves relevant history for context
   - limits conversation length to manage memory

3. **config**: handles environment variables and settings
   - validates required configuration
   - provides default timeouts and retries

4. **discordbot**: main bot implementation
   - handles discord interactions
   - implements slash commands
   - manages user permissions
   - coordinates between other components

### data flow
1. user sends a question via mention or `/query`
2. bot retrieves conversation history for that user
3. bot searches documents for relevant information
4. combined context is sent to LLM via openrouter
5. ai response is delivered to the user
6. conversation history is updated

---

## troubleshooting

### bot doesn't respond to commands
- verify your discord bot token is correct
- ensure bot has required permissions in your server
- check that command sync was successful (see logs)

### ai responses are poor quality
- check that documents are properly added to the knowledge base
- verify your openrouter API key is valid
- consider adjusting the system prompt to better suit your needs

### document search not finding relevant information
- add more documents with comprehensive information
- break large documents into smaller, focused files
- use more specific queries

### common error messages
- "failed to sync commands": discord API connectivity issue
- "error adding document": problem with document format or storage
- "error processing query": issue with LLM or API access

---

## developer reference

### adding new commands

to add a new command, add to the `setup_commands` method in the `DiscordBot` class:

```python
@self.tree.command(name="your_command", description="Command description")
@app_commands.describe(param="Parameter description")
async def your_command(interaction: discord.Interaction, param: str):
    await interaction.response.defer()
    try:
        # your command logic here
        await interaction.followup.send("Response")
    except Exception as e:
        await interaction.followup.send(f"Error: {str(e)}")
```

### modifying system prompt

edit the `system_prompt.py` file to change how publicia responds to queries.

### extending document types

the `DocumentManager._load_documents` method can be extended to support additional file types beyond .txt files.

### custom embeddings

to use a different embedding model, modify the model initialization in `DocumentManager.__init__`:

```python
self.model = SentenceTransformer('your-preferred-model')
```

---

*documentation for publicia v2.0 - your neural interface to ledus banum lore!*
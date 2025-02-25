# Publicia Bot Documentation

*your enhanced neural interface to ledus banum 77 and imperial lore with advanced query processing!*

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

publicia is a discord bot that serves as an intelligent neural interface to your lore documents. it uses advanced embedding technology and a multi-stage AI pipeline to understand and respond to questions about your setting, maintaining personalized conversation history with each user.

### key capabilities
- answers questions about ledus banum 77 and imperial lore with high accuracy
- employs a sophisticated five-stage query processing pipeline
- uses gemini for query analysis and document synthesis
- generates responses with deepseek-r1 for consistent character voice
- remembers conversations with individual users
- imports documents from text files or google docs
- uses semantic search across all knowledge documents

---

## setup & installation

### prerequisites
- python 3.8 or higher
- discord bot token (from discord developer portal)
- openrouter API key with access to:
  - google/gemini-2.0-flash-001 (for query analysis)
  - deepseek/deepseek-r1 (for response generation)
- access to multiple language models (via openrouter)

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
LLM_MODEL=deepseek/deepseek-r1
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
python PubliciaV4.py
```

---

## configuration

publicia uses environment variables for configuration:

| variable | required | default | description |
|----------|----------|---------|-------------|
| DISCORD_BOT_TOKEN | yes | none | token for discord bot authentication |
| OPENROUTER_API_KEY | yes | none | API key for openrouter service |
| LLM_MODEL | no | deepseek/deepseek-r1 | model for final response generation |
| CLASSIFIER_MODEL | no | google/gemini-2.0-flash-001 | model for query analysis and document synthesis |
| API_TIMEOUT | no | 30 | timeout in seconds for API calls |
| MAX_RETRIES | no | 10 | maximum retry attempts for API calls |

### model configuration

publicia uses two configurable AI models through the openrouter API:

1. **CLASSIFIER_MODEL**: Used for query analysis and document synthesis (default: google/gemini-2.0-flash-001)
2. **LLM_MODEL**: Used for final response generation in publicia's character voice (default: deepseek/deepseek-r1)

the system includes a dynamic fallback system that adapts based on the model family:

```
if a deepseek model is requested:
  - tries variants like model:floor, model:nitro
  - falls back to other deepseek models
  - ultimately tries gemini models

if a google model is requested:
  - tries other gemini variants
  - falls back to deepseek models

general fallback sequence is constructed dynamically based on the requested model
```

you can configure both models separately in your .env file:
```
# Use Claude for responses but keep Gemini for analysis
LLM_MODEL=anthropic/claude-3-haiku-20240307
CLASSIFIER_MODEL=google/gemini-2.0-flash-001

# Or use the same model for everything
LLM_MODEL=anthropic/claude-3-opus-20240229
CLASSIFIER_MODEL=anthropic/claude-3-opus-20240229
```

---

## features

### enhanced query pipeline
publicia now uses a sophisticated five-stage pipeline to process user questions:
1. **query analysis**: gemini analyzes the question to extract key topics and search terms
2. **enhanced search**: uses extracted keywords to perform targeted semantic search
3. **document synthesis**: gemini organizes search results into coherent context
4. **context assembly**: combines conversation history, synthesized context, and raw references
5. **response generation**: deepseek-r1 generates the final in-character response

### semantic document search
publicia uses embeddings to understand the meaning behind user questions, not just keywords! this allows for more natural queries and better answers, enhanced by gemini's query analysis.

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
| /query | ask publicia a question (using enhanced pipeline) | `/query question:what is the capital of the empire?` |
| /listcommands | show all available commands | `/listcommands` |
| /lobotomise | clear your conversation history | `/lobotomise` |

### document management
| command | description | usage |
|---------|-------------|-------|
| /add_info | add a document to the knowledge base | `/add_info name:empire_history content:long ago...` |
| /add_doc | add a document via file attachment | `!add_doc name` with attached .txt file |
| /listdocs | list all available documents | `/listdocs` |
| /removedoc | remove a document | `/removedoc name:outdated_lore` |
| /searchdocs | search documents directly | `/searchdocs query:emperor's throne` |

### google docs integration
| command | description | usage |
|---------|-------------|-------|
| /add_googledoc | track a google doc | `/add_googledoc doc_url:https://docs.google.com/... name:imperial_navy` |
| /list_googledocs | show tracked google docs | `/list_googledocs` |
| /remove_googledoc | remove a tracked google doc | `/remove_googledoc identifier:imperial_navy` |

### moderation (admin only)
| command | description | usage |
|---------|-------------|-------|
| /ban_user | ban user from using the bot | `/ban_user user:@troublemaker` |
| /unban_user | unban a previously banned user | `/unban_user user:@reformed_user` |

### special message functions
- mention the bot to ask questions directly (uses enhanced pipeline)
- include "LOBOTOMISE" in a message to clear your history

---

## architecture

publicia consists of six main components:

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

5. **query processor**: handles the enhanced query pipeline
   - analyze_query(): uses gemini to extract key search terms
   - enhanced_search(): performs targeted document search
   - synthesize_results(): uses gemini to organize search results

6. **model manager**: handles AI model access
   - _try_ai_completion(): manages model selection and fallbacks
   - prioritizes models based on task requirements

### enhanced query pipeline

the query pipeline processes user questions in five stages:

1. **query analysis** (gemini)
   - extracts main topic and important keywords
   - identifies entity types and relevant document categories
   - suggests search strategy

2. **enhanced document search**
   - enriches query with extracted keywords
   - performs semantic search with sentence transformers
   - ranks document chunks by relevance

3. **document synthesis** (gemini)
   - organizes search results into coherent structure
   - highlights connections between information pieces
   - identifies contradictions or gaps

4. **context assembly**
   - combines conversation history
   - includes synthesized document context
   - adds raw document chunks with citation links
   - preserves channel and user information

5. **response generation** (deepseek-r1)
   - generates final response in publicia's character voice
   - refers to provided context for information
   - maintains consistent roleplay style

### data flow
1. user sends a question via mention or `/query`
2. gemini analyzes the query to extract key search terms
3. bot searches documents based on enhanced query terms
4. gemini synthesizes search results into coherent context
5. combined context is sent to deepseek-r1 via openrouter
6. ai response is delivered to the user
7. conversation history is updated

---

## troubleshooting

### bot doesn't respond to commands
- verify your discord bot token is correct
- ensure bot has required permissions in your server
- check that command sync was successful (see logs)

### ai responses are poor quality
- check that documents are properly added to the knowledge base
- verify your openrouter API key is valid
- ensure you have access to both gemini and deepseek models
- consider adjusting the system prompt to better suit your needs

### query analysis not working properly
- check openrouter access to gemini-2.0-flash-001
- verify log messages for analysis stage completion
- try increasing the timeout duration in config
- ensure json parsing is working correctly

### document search not finding relevant information
- add more documents with comprehensive information
- break large documents into smaller, focused files
- use more specific queries
- check document embeddings are generating correctly

### common error messages
- "error analyzing query": issue with gemini access or processing
- "error synthesizing results": problem with document synthesis
- "all models failed to generate completion": openrouter connection issues
- "neural circuit overload": general error in the pipeline processing

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

### enhanced query pipeline methods

key methods for the enhanced query pipeline:

```python
# Query analysis with Gemini
async def analyze_query(self, query: str) -> Dict:
    """Use Gemini to analyze the query and extract keywords/topics."""
    # Implementation details...

# Enhanced document search
async def enhanced_search(self, query: str, analysis: Dict) -> List[Tuple[str, str, float]]:
    """Perform an enhanced search based on query analysis."""
    # Implementation details...

# Document synthesis with Gemini
async def synthesize_results(self, query: str, search_results: List[Tuple[str, str, float]], analysis: Dict) -> str:
    """Use Gemini to synthesize search results into a coherent context."""
    # Implementation details...
```

### modifying the model fallback order

to change which models are tried and in what order, modify the `_try_ai_completion` method:

```python
# For DeepSeek-R1 responses, prioritize these models
models = [
    "deepseek/deepseek-r1",           # primary model
    "deepseek/deepseek-r1:floor",     # first fallback
    # Add or remove models here
]
```

---

*documentation for publicia v4.0 - your neural interface to ledus banum lore with enhanced ai pipeline!*
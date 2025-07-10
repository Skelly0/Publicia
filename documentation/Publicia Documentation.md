# Publicia Bot Documentation

*A hybrid RAG (Retrieval-Augmented Generation) Discord bot using vector embeddings, BM25 search, and multi-model LLM integration for intelligent document querying and conversational AI.*

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup Guide](#setup-guide)
- [Command Reference](#command-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Recent Updates](#recent-updates)

## Overview

Publicia is a Discord bot designed to serve as an interactive lore repository for the fictional setting of Ledus Banum 77 and the Infinite Empire, the setting of DPS Season 7. The bot takes on the character of Publicia, an "Imperial abhuman mentat", allowing users to query information through natural conversation.

### Key Capabilities

- **Document Search & Retrieval**: Uses an advanced hybrid system (vector embeddings + BM25 keyword matching) with AI-generated contextual retrieval and sophisticated reranking for highly accurate information retrieval. Documents are managed internally using unique UUIDs.
- **Image Analysis**: Can process, store, and analyze images related to the lore using vision-capable models.
- **Conversation Memory**: Remembers conversation history for contextual responses and context-aware searching.
- **Channel Context Parsing**: Optionally includes recent messages from the current channel as general context in the AI prompt, configurable via the `/parse_channel` command.
- **Multiple AI Models**: Supports various AI models with automatic fallback and retry mechanisms.
- **Google Doc Integration**: Can fetch and index content from Google Docs (linking them to internal document UUIDs), automatically detecting changes using content hashing. Also supports designating a specific channel where posted Google Doc links will be automatically added/refreshed.
- **Google Sheets Integration**: Supports tracking specific tabs from Google Sheets, converting rows to readable text and refreshing them automatically.
- **Role-Playing**: Maintains character as Publicia while providing information.
- **File Management**: Allows listing and retrieving stored documents (identified by UUIDs).
- **DOCX Processing**: Can process `.docx` files to automatically tag specific colored text (e.g., #980000) with XML tags like `<post-invasion_lore>` for specialized lore integration.

### Lore Context

The bot is designed to role-play as Publicia, an abhuman mentat specializing in knowledge about Ledus Banum 77 (also known as Tundra) and the Infinite Empire. This is for DPS Season 7. The character adheres to specific lore elements.

## Architecture

Publicia is built on Python using discord.py, with several specialized components working together:

### System Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Discord Bot    │◄────┤ Conversation Mgr │◄────┤  User Prefs     │
│  (Commands)     │     │ (Memory System)  │     │  (Model Choice) │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │                        ▲
         ▼                        │
┌────────────────┐      ┌─────────┴──────────┐
│ Document Mgr   │◄─────┤     AI Backend     │
│ (Vector+BM25)  │      │ (OpenRouter API)   │
└────────┬────────┘      └──────────────────┘
         │                        ▲
         ▼                        │
┌────────────────┐                │
│  Image Mgr     │────────────────┘
│ (Storage/OCR)  │
└────────────────┘
```

### Key Classes

1. **DiscordBot**: Main bot class that handles Discord integration and commands
2. **DocumentManager**: Manages document storage (using UUIDs as primary keys), embedding generation (including contextualized embeddings), BM25 indexing, hybrid search (vector + BM25 with score fusion), contextual retrieval, reranking, and handles migration from older name-based document systems.
3. **ImageManager**: Handles image storage, retrieval, and description generation
4. **ConversationManager**: Maintains conversation history for users
5. **UserPreferencesManager**: Handles user preferences like AI model selection
6. **Config**: Loads and provides configuration settings

### Data Flow

1. User sends a query (message or command)
2. Bot determines if the query is context-dependent (follow-up) and enhances it with conversation history if needed.
3. Performs a hybrid search using context-aware embeddings (if applicable), combining semantic similarity (vectors) and keyword matching (BM25) via score-based fusion.
4. Search results, including AI-generated contextual enhancements for each chunk, are prepared as context.
5. User's conversation history is added for continuity
6. Recent channel messages (if parsing is enabled for the channel) are added as general context.
7. AI model generates response based on all context
8. Response is sent back to user, potentially with images
8. Conversation history is updated

## Features

### Document Management & Hybrid Search

The bot uses an advanced hybrid search system combining vector embeddings and BM25 keyword matching:

- Documents are chunked into smaller sections for precise retrieval and identified internally by unique UUIDs.
- **Contextual Retrieval**: Before embedding, each chunk is enhanced with AI-generated context explaining its relationship to the whole document. This significantly improves relevance.
- Contextualized chunks are converted to vector embeddings using Google's Generative AI (`models/text-embedding-004`).
- **BM25 Indexing**: A separate BM25 index is maintained for efficient keyword matching.
- **Hybrid Search Execution**: Queries are processed using both the vector embeddings (for semantic meaning) and the BM25 index (for keywords).
- **Score-Based Fusion**: Results from both search methods are combined using a weighted scoring system (e.g., 60% embedding score, 40% BM25 score) to produce a final relevance ranking.
- Queries are matched to the most relevant document chunks.
- Similarity scores determine the best matches.
- Documents can be added (assigned a UUID), removed (by UUID), or searched directly.

### DOCX Lore Processing

Publicia includes a feature to process `.docx` files, specifically looking for text colored with `#980000` (a dark red). This text is then wrapped with `<post-invasion_lore>` XML-like tags.
- This is handled by the `tag_lore_in_docx` function within the `DocumentManager`.
- The purpose is to allow for easy marking and extraction/identification of specific lore sections within Word documents.
- This functionality requires the `python-docx` library to be installed in the bot's environment.
- The processed text (with tags) is typically saved as a new `.txt` document within Publicia's knowledge base.

### Contextual Retrieval (Integrated into Document Management)

This sophisticated approach improves search accuracy and content understanding:

- AI generates context for each document chunk to explain its relationship to the whole document
- Context is prepended to chunks before embedding and indexing
- The same contextualized chunks are included in prompts sent to the model
- This solves problems with ambiguous pronouns, missing entity references, and lack of temporal context
- Significantly reduces retrieval failures by up to 49% based on research
- Uses Gemini 2.5 Flash via OpenRouter to generate context

### Image Processing

The bot can work with images in multiple ways:

- Store images with descriptions in the knowledge base
- Auto-generate descriptions using vision-capable models
- Include relevant images in search results
- Process images attached to messages
- Support basic image management operations

### Conversation Memory

The bot maintains conversation history for each user:

- Stores both user and bot messages
- Uses history to provide context for future queries
- Supports viewing, managing, and deleting conversation history
- Implements a lobotomise and memory_clear command to wipe history
- JSON-based storage for persistence

### Google Doc Integration

Unique capability to work with Google Docs:

- Track Google Docs with custom names. These are linked internally to a unique document UUID within Publicia.
- Automatically refresh content on a schedule.
- **Efficient Change Detection**: Uses content hashing to compare the current document content with the stored version, only processing and re-indexing if actual changes are detected.
- **Automatic Tracking Channels**: Use `/set_doc_channel` to add one or more channels where any Google Doc links posted will be automatically tracked. The bot will react to indicate success (✅), partial success (⚠️), or failure (❌).
- Extract content from Google Doc links in messages
- Create citations linking back to source documents
- Automatically decodes HTML entities (like `&`) found in Google Doc titles for cleaner display.
- Support renaming and removing tracked documents (which updates the link to the internal UUID).

### Google Sheets Integration

Similar support exists for Google Sheets:

- Track specific tabs from a sheet using the URL and tab name.
- Rows are converted to `Header: value` lines for easy reading.
- Sheets refresh on the same schedule as Google Docs.

### AI Model Selection

Users can choose their preferred AI model:

- **Gemini 2.5 Flash**: Optimized for accuracy, image analysis, and thinking capabilities
- **Qwen QwQ 32B**: Great for roleplaying with strong lore accuracy
- **Qwen 3 235B A22B**: Large parameter model with enhanced capabilities
- **DeepSeek V3 0324**: Advanced conversational model with improved performance
- **DeepSeek-R1**: Best for immersive roleplaying and reasoning
- **Claude 3.5 Haiku**: Fast responses with image capabilities
- **Claude 4 Sonnet**: Premium capabilities (admin restricted)
- **Nous: Hermes 405B**: Balanced between creativity and precision
- **Wayfarer 70B**: Optimized for narrative-driven roleplay
- **Grok 3 Mini**: X.AI's efficient model for quick responses

Additional models available to administrators include Gemini 2.5 Pro, Claude 3.7 Sonnet, Anubis Pro 105B, Llama 4 Maverick, OpenAI GPT-4.1 models, and Phi-4 Multimodal.

Each model has different strengths, fallback mechanisms ensure reliability.

### Search and Retrieval Process

The enhanced search system follows these steps:

1.  **Context Dependence Check**: Determines if the query is a follow-up question requiring conversation history.
2.  **Query Enhancement (if needed)**: Rewrites the query using conversation context for better relevance.
3.  **Context-Aware Embedding Generation (if needed)**: Creates a query embedding that incorporates conversation context.
4.  **Hybrid Search**: Executes both embedding-based semantic search and BM25 keyword search using the (potentially context-enhanced) query.
5.  **Score Fusion**: Combines results from both search methods using weighted scores (e.g., 60% embedding, 40% BM25).
6.  **Result Caching**: Stores search results to speed up potential follow-up queries.
7.  **Reranking (Optional)**: Applies a sophisticated reranking algorithm to the fused results for further refinement, with a fallback mechanism.
8.  **Contextual Enhancement**: Retrieves the AI-generated context associated with each resulting document chunk.
9.  **Prompt Assembly**: Constructs the final prompt for the LLM, including the contextualized chunks, conversation history, and the original query.
10. **Response Generation**: The LLM generates a response based on the provided context.
11. **Empty Response Handling**: Includes an automatic retry system if the LLM returns a blank or very short response.

### Embedding System

The bot uses Google's Generative AI for embeddings:

- Uses the `models/text-embedding-004` embedding model
- Supports configurable embedding dimensions
- Automatically handles truncation for storage efficiency
- Provides better semantic understanding than previous embedding systems

### Hybrid Search System Details

The core search mechanism combines multiple techniques:

-   **Vector Embeddings**: Uses Google's `models/text-embedding-004` for semantic understanding (default 60% weight in fusion). Embeddings are generated on *contextualized* chunks.
-   **BM25 (Best Matching 25)**: Provides robust keyword-based matching (default 40% weight in fusion). Operates on the original chunk text.
-   **Score-Based Fusion**: Merges results from vector and BM25 searches using normalized, weighted scores for a comprehensive ranking.
-   **Context-Aware Search**: Detects follow-up questions and automatically:
    -   Enhances the query with relevant conversation history.
    -   Generates a *context-aware embedding* for the enhanced query.
-   **Search Result Caching**: Stores results from a query to efficiently provide additional, non-overlapping results for subsequent related queries (e.g., "tell me more").

### Dynamic Response Temperature

Automatically adjusts the LLM temperature based on query type:

- Low temperature (TEMPERATURE_MIN) for factual/information queries
- Higher temperature (TEMPERATURE_MAX) for creative/roleplay scenarios
- Base temperature (TEMPERATURE_BASE) for balanced queries
- Detects query context through analysis of roleplay elements, question markers, etc.

### Result Reranking

Sophisticated reranking of search results to improve relevance:

- Multiple filter modes: strict, dynamic, and topk
- Customizable minimum score threshold
- Reranking candidates configuration
- Weighted combination of initial and reranked scores
- Fallback mechanism to ensure results are always returned
- TOP_K now acts as a maximum limit for chunks parsed

## Setup Guide

### Prerequisites

- Python 3.8+ with pip
- Discord Bot Token (from Discord Developer Portal)
- OpenRouter API Key
- Google API Key (for embeddings generation)
- Required Python packages (see requirements section)

### Installation Steps

1. Clone the repository or download the source files
2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with configuration
```
# Required
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_api_key
GOOGLE_API_KEY=your_google_api_key

# Optional (defaults shown)
LLM_MODEL=google/gemini-2.5-flash-preview # Default model for text generation
DEFAULT_MODEL=qwen/qwq-32b # Default user preference model
CLASSIFIER_MODEL=google/gemini-2.5-flash-preview # Model for classification tasks
EMBEDDING_MODEL=models/text-embedding-004 # Model for document embeddings
EMBEDDING_DIMENSIONS=0 # Set to positive number to truncate embeddings (0 = no truncation)

# Chunk configuration
CHUNK_SIZE=300 # Words per chunk
CHUNK_OVERLAP=30 # Words overlap between chunks

# Search configuration
TOP_K=5 # Base number of search results
MAX_TOP_K=20 # Maximum search results
TOP_K_MULTIPLIER=1.0 # Multiplier for TOP_K
BM25_WEIGHT=0.25 # Weight for BM25 vs embedding search (0.25 = 25% BM25, 75% embedding)

# API settings
API_TIMEOUT=180 # Seconds
MAX_RETRIES=10

# Temperature settings for different query types
TEMPERATURE_MIN=0.0 # For factual queries
TEMPERATURE_BASE=0.1 # Default balanced temperature
TEMPERATURE_MAX=0.4 # For creative/roleplay queries

# Reranking configuration
RERANKING_ENABLED=false # Enable/disable result reranking
RERANKING_CANDIDATES=20 # Number of initial candidates for reranking
RERANKING_MIN_SCORE=0.5 # Minimum relevance score threshold
RERANKING_FILTER_MODE=strict # Options: strict, dynamic, topk

# Contextualization settings
CONTEXTUALIZATION_ENABLED=true # Enable AI-generated context for chunks
MAX_WORDS_FOR_CONTEXT=20000 # Maximum words to consider for context generation
USE_CONTEXTUALISED_CHUNKS=true # Use contextualized chunks in prompts
CHANNEL_CONTEXTUALIZATION_ENABLED=true # Contextualize channel archives

# Feature toggles
KEYWORD_DATABASE_ENABLED=true # Enable keyword database system
KEYWORD_CHECK_CHUNK_LIMIT=5 # Number of chunks to check for keywords
DOCUMENT_LIST_ENABLED=true # Include internal document list in LLM prompts

# Optional: Auto-process Google Docs with lore tagging
AUTO_PROCESS_GOOGLE_DOCS=false # Set to true to enable automatic .docx download and processing

# Automatic Google Doc tracking channels are stored in
# `documents/doc_tracking_channels.json`

# Optional: Permission settings (comma-separated user/role IDs)
# ALLOWED_USER_IDS=123456789,987654321
# ALLOWED_ROLE_IDS=111111111,222222222
```

5. Create necessary directories
```bash
mkdir -p documents images conversations user_preferences lorebooks
```

### Required Packages

```
discord.py>=2.3.1
python-dotenv>=1.0.0
sentence-transformers>=2.2.2
numpy>=1.22.3
aiohttp>=3.8.5
apscheduler>=3.10.1
torch>=2.0.0
pathlib>=1.0.1
regex>=2023.0.0
uuid>=1.30
deque>=1.0
google-generativeai>=0.3.0
rank_bm25>=0.2.2
```
Note: Added `rank_bm25` for BM25 functionality.

### Running the Bot

```bash
python PubliciaV13.py
```

The bot will display a startup banner and initialize all components.

## Command Reference

### Slash Commands

#### Document Management

- `/add_info`: Add text directly to the knowledge base. A unique UUID will be assigned to the new document. **(Admin Only)**
  - **Parameters**: `name` (Original name for the document), `content` (Text content of the document).

- `/list_docs`: Show all documents in the knowledge base, sorted alphabetically by their original name. The list includes the original name and its corresponding UUID.

- `/remove_doc`: Remove a document from the knowledge base using its UUID. **(Admin Only)**
  - **Parameters**: `doc_uuid` (The UUID of the document to remove).

- `/search_docs`: Search documents directly based on content.
  - **Parameters**: `query` (The search term or question).
- `/search_keyword`: Search for a specific keyword within documents.
  - **Parameters**: `keyword` (The word to search for).
- `/search_keyword_bm25`: Search for a keyword using BM25 ranking.
  - **Parameters**: `keyword` (The word to search for).

- `/add_googledoc`: Track a Google Doc, linking it to an internal document UUID. **(Admin Only)**
  - **Parameters**: `doc_url` (The URL or ID of the Google Doc), `name` (Optional custom original name for this document in Publicia).

- `/list_googledocs`: List all tracked Google Docs, showing their original names, Google Doc IDs, and internal Publicia UUIDs.

- `/remove_googledoc`: Remove a tracked Google Doc. This removes the tracking entry and the associated local document (by its internal UUID). **(Admin Only)**
  - **Parameters**: `identifier` (The Google Doc ID, URL, or its custom original name in Publicia).
- `/add_googlesheet`: Track a specific tab from a Google Sheet. **(Admin Only)**
  - **Parameters**: `sheet_url` (URL or ID of the sheet), `tab` (Tab name), `name` (Optional custom name), `header_row` (Row number containing headers, default: 1).
- `/list_googlesheets`: List all tracked Google Sheets.
- `/remove_googlesheet`: Remove a tracked Google Sheet. **(Admin Only)**
  - **Parameters**: `identifier` (Sheet ID or custom name), `tab` (Tab name).

- `/rename_document`: Rename the user-facing original name of a document. The document is identified by its UUID or current original name. **(Admin Only)**
  - **Parameters**: `current_name` (The document's current UUID or original name), `new_name` (The new original name for the document).

- `/retrieve_file`: Retrieve a document from Publicia's storage by its UUID.
  - **Parameters**: `doc_uuid` (The UUID of the document to retrieve).
  - Downloads the requested file as a Discord attachment, using its original name.

- `/archive_channel`: Archive messages from a specified Discord channel into a new document in the knowledge base. A UUID will be assigned to the new document(s). **(Admin Only)**
  - **Parameters**:
    - `channel`: The Discord text channel to archive.
    - `message_limit`: Maximum number of messages to fetch (0 for all, default: 1000, max: 10000).
    - `document_name`: Optional name for the saved document (default: `channel_archive_[channel_name]_[timestamp]`).
    - `include_bots`: Whether to include messages from bots (default: True).
    - `include_attachments`: Whether to include attachment URLs in the archive (default: True).

- `/summarize_doc`: Generate a concise summary of a document using an LLM. The document can be specified by its UUID or original name.
  - **Parameters**: `identifier` (The UUID or original name of the document to summarize).

- `/view_chunk`: View the content of a specific chunk from a document. The document can be specified by its UUID or original name.
  - **Parameters**:
    - `identifier`: The UUID or original name of the document.
    - `chunk_index`: The 1-based index of the chunk to view.
    - `contextualized`: (Optional, default: False) Whether to view the contextualized version of the chunk (used for embeddings) or the original chunk text.

- `/process_docx_lore`: Process an uploaded `.docx` file to identify text colored with `#980000` and wrap it with `<post-invasion_lore>` XML-like tags. The result is typically saved as a new `.txt` document. Requires the `python-docx` library. **(Admin Only)**
  - **Parameters**:
    - `docx_file`: The `.docx` file to process (as a Discord attachment).
    - `output_filename`: (Optional) Custom name for the output `.txt` file. If not provided, defaults to `[input_filename_stem]_processed.txt`.

- `/set_doc_channel`: Add a channel for automatic Google Doc tracking. **(Admin Only)**
  - **Parameters**: `channel` (The text channel to monitor).

- `/reload_docs`: Reload all documents from disk, re-processing them. **(Admin Only)**

- `/regenerate_embeddings`: Regenerate all document embeddings. **(Admin Only)**

- `/refresh_docs`: Manually refresh all tracked Google Docs (checks for changes based on content hashes). **(Admin Only)**

- `/force_refresh_googledocs`: Force refresh and re-process ALL tracked Google Docs (bypasses change detection). **(Admin Only)**
- `/refresh_sheets`: Manually refresh all tracked Google Sheets. **(Admin Only)**
- `/force_refresh_googlesheets`: Force refresh and re-process ALL tracked Google Sheets. **(Admin Only)**

- `/track_channel`: Start tracking a Discord channel and archive it periodically. The bot will create an archive document and update it every 6 hours (configurable) with new messages. **(Admin Only)**
  - **Parameters**:
    - `channel`: The Discord text channel to track.
    - `update_interval_hours`: How often to check for updates (default: 6 hours).

- `/untrack_channel`: Stop tracking a Discord channel. The archive document will no longer be updated, but existing archives remain in the knowledge base. **(Admin Only)**
  - **Parameters**: `channel`: The Discord text channel to stop tracking.

#### Image Management

- `/list_images`: Show all images in knowledge base
- `/view_image`: View an image
  - **Parameters**: `image_id` (ID of image)
- `/edit_image`: View and edit an image description **(Admin Only)**
  - **Parameters**: `image_id` (ID of image)
- `/remove_image`: Remove an image **(Admin Only)**
  - **Parameters**: `image_id` (ID of image)
- `/update_image_description`: Update image description **(Admin Only)**
  - **Parameters**: `image_id` (ID of image), `description` (new description)

#### Query and Conversation / Context Management

- `/query`: Ask a question with optional image
  - **Parameters**: `question` (your query), `image_url` (optional)

- `/query_full_context`: Ask a question using ALL documents as context (1/day limit)
  - **Parameters**: `question` (your query)
  - Uses powerful models like Gemini 2.5 Pro for comprehensive analysis.

- `/history`: View your conversation history
  - **Parameters**: `limit` (number of messages, default 10)

- `/manage_history`: View and manage history with indices
  - **Parameters**: `limit` (number of messages)

- `/delete_history_messages`: Delete specific messages
  - **Parameters**: `indices` (comma-separated list), `confirm` (must be "yes")

- `/lobotomise`: Wipe your conversation history (Admin can wipe others' history using the `user` parameter) **(Admin Only)**
  - **Parameters**: `user` (optional, user to lobotomise)
- `/memory_clear`: Clear the bot's short-term memory for the current conversation (Admin can clear for others) **(Admin Only)**
  - **Parameters**: `user` (optional, user whose memory to clear)
- `/delete_history_messages`: Delete specific messages from history (Admin can delete for others) **(Admin Only)**
  - **Parameters**: `indices` (comma-separated list), `confirm` (must be "yes"), `user` (optional, user whose history to modify)
- `/archive_conversation`: Archive a user's conversation history **(Admin Only)**
  - **Parameters**: `user` (The user whose history to archive), `archive_name` (Name for the archive)
- `/delete_archive`: Delete a conversation archive **(Admin Only)**
  - **Parameters**: `archive_name` (Name of the archive)

#### Settings and Utilities

- `/set_model`: Change your preferred AI model
  - **Parameters**: `model` (model choice from options)

- `/get_model`: Show your current model

- `/toggle_debug`: Toggle showing model info in responses

- `/list_commands`: Show all available commands, categorized by function and indicating admin-only status.
- `/help`: Show a succinct overview of capabilities and direct to `/list_commands` for details
- `/parse_channel`: Toggle parsing of recent channel messages to be included as general context in AI prompts
  - **Parameters**: `enabled` (true/false), `message_count` (number of messages, 1-200)
- `/ban_user`: Ban a user from using the bot **(Admin Only)**
  - **Parameters**: `user` (Discord user)
- `/unban_user`: Unban a user **(Admin Only)**
  - **Parameters**: `user` (Discord user)
- `/compare_models`: Compare responses from multiple AI models **(Admin Only)**
  - **Parameters**: `question`, `model_types` (optional), `max_models` (optional), `image_url` (optional), `private` (optional)

### Prefix Commands

These commands use the prefix "Publicia!" instead of slash commands. They are generally older commands and many are now restricted to administrators.

#### Admin Only Prefix Commands

- `Publicia! add_doc "Document Name"`: Add document (with optional attachment). **Requires permissions.**
- `Publicia! add_image "Image Name" [yes/no]`: Add image with optional auto-description. **Requires permissions.**

## Advanced Usage

### Model Selection Strategy

Choose models based on your needs:

- **Gemini 2.5 Flash**: Best for factual accuracy, citations, image analysis, and thinking capabilities
- **Qwen QwQ 32B**: Great for roleplaying with strong lore accuracy and reasoning
- **Qwen 3 235B A22B**: Large parameter model for complex tasks requiring extensive knowledge
- **DeepSeek V3 0324**: Advanced conversational model with improved performance and efficiency
- **DeepSeek-R1**: Use for immersive, creative responses and advanced reasoning
- **Claude 3.5 Haiku**: Fast responses with image capabilities and good efficiency
- **Claude 4 Sonnet**: Advanced capabilities with enhanced creativity (admin-only)
- **Nous: Hermes 405B**: Good middle-ground between creativity and precision
- **Wayfarer 70B**: Optimized for narrative-driven roleplay with realistic stakes and conflicts
- **Grok 3 Mini**: X.AI's efficient model for quick, conversational responses

**Admin-only models** include Gemini 2.5 Pro, Claude 3.7 Sonnet, Anubis Pro 105B, Llama 4 Maverick, OpenAI GPT-4.1 models, and Phi-4 Multimodal for specialized use cases.

The bot will automatically fall back to available models if your preferred model fails, and includes an automatic retry system when models return blank or extremely short responses.

### Document Organization Tips

- Keep document names descriptive but concise
- Group related information in single documents
- Use Google Docs for frequently updated information
- Add citations in documents when possible
- Images with detailed descriptions improve search relevance

### Performance Optimization

- Keep conversation histories manageable
- Limit number of tracked Google Docs
- Use specific queries for better search precision
- Consider removing unused documents to reduce search space
- Choose faster models (like Gemini Flash) for simple queries
- Set EMBEDDING_DIMENSIONS to truncate embeddings for better storage efficiency
- Configure dynamic temperature settings for response quality and relevance
- Optimize reranking settings based on your needs

### Hybrid Search Configuration

Configure the hybrid search system for optimal performance:

-   **Fusion Weights**: Adjust the balance between semantic (embedding) and keyword (BM25) search in the `.env` file (defaults usually 0.6 for embedding, 0.4 for BM25).
-   **Reranking Settings**: Modify `RERANKING_ENABLED`, `RERANKING_CANDIDATES`, `RERANKING_MIN_SCORE`, and `RERANKING_FILTER_MODE` in `.env` to fine-tune result filtering.
-   **Contextual Retrieval Prompt**: While not directly configurable via `.env`, the prompt used for generating chunk context (`prompts/context_prompt.py` - *verify path*) influences retrieval. Modifications require code changes.
-   **Context-Aware Search**: This is largely automatic but relies on good conversation history management.

### Contextual Retrieval Best Practices

-   **Document Structure**: Keep original documents well-structured with clear headings and logical flow, as this aids the AI in generating useful context for chunks.
-   **Regeneration**: Remember that context is generated *once* when a document is added or updated. If you significantly restructure a document or the context generation logic improves, use `/regenerate_embeddings` for the specific document (or all documents) to apply the changes.

### Backup Strategies

- Regularly back up the following directories:
  - `documents/`
  - `images/`
  - `conversations/`
  - `user_preferences/`
  - `lorebooks/`

- Consider exporting Google Docs as local backups periodically

### Image Analysis Best Practices

- Provide detailed descriptions when adding images manually
- Use vision-capable models (Gemini, Claude) when working with images
- Label and tag images appropriately
- For complex images, add multiple versions with different aspects highlighted

### Debug Tools

-   **`/toggle_debug`**: Shows the specific AI model used for each response and potentially other metadata.
-   **Logs**: Check `bot_detailed.log` for detailed operational information, errors, and search process steps.
-   **Admin Commands**: `/reload_docs`, `/regenerate_embeddings` are crucial for maintenance and applying configuration changes.

### Custom Temperature Settings

The bot uses dynamic temperature control based on query type:

- Edit `TEMPERATURE_MIN` (default: 0.0) for factual queries
- Edit `TEMPERATURE_BASE` (default: 0.1) for balanced queries
- Edit `TEMPERATURE_MAX` (default: 0.4) for creative roleplay
- You can override these values per user with the `/temperature` command (values must satisfy `0.0 ≤ min ≤ base ≤ max ≤ 2.0`; call with no values to reset to defaults)
- The bot automatically analyzes queries and conversation context to select an appropriate temperature

### Reranking Configuration

Customize search result reranking with these settings:

- RERANKING_ENABLED: Turn reranking on/off
- RERANKING_CANDIDATES: Number of initial results to consider for reranking
- RERANKING_MIN_SCORE: Threshold score for relevance
- RERANKING_FILTER_MODE: Choose between strict (absolute threshold), dynamic (adapts threshold based on results), or topk (traditional top-k regardless of score)

## Troubleshooting

### Common Issues

#### Bot Not Responding
- Ensure Discord bot token is correct
- Check that the bot has proper permissions in Discord
- Verify internet connection and API access

#### Search Not Finding Relevant Results
-   **Query Specificity**: Try more specific queries.
-   **Document Content**: Ensure relevant information exists and is well-structured in the knowledge base.
-   **Hybrid Search Weights**: Adjust the embedding vs. BM25 weights in `.env` if results seem skewed towards semantic or keyword matches inappropriately.
-   **Reranking Settings**: Experiment with `RERANKING_MIN_SCORE` and `RERANKING_FILTER_MODE`. A `strict` filter might be too aggressive, while `topk` might return irrelevant results.
-   **Embeddings**: Consider regenerating embeddings (`/regenerate_embeddings`) if you suspect issues or after significant code changes.

#### Model Errors
- Ensure OpenRouter API key is valid
- Check if chosen model is available
- Try a different model
- Verify request format and size limits

#### Google Doc Integration Issues
- Ensure Docs are publicly accessible
- Check that Doc IDs are correct
- Verify internet connection
- Try removing and re-adding the Doc
- Verify the content hash mechanism is working correctly

#### Image Processing Failures
- Check supported image formats (PNG, JPG, etc.)
- Ensure image URLs are direct links to images
- Try using a different vision-capable model
- Reduce image size if very large

#### Embedding Issues
- Ensure Google API key is valid and has access to embedding models
- Check if EMBEDDING_DIMENSIONS is set correctly
- Try regenerating embeddings with `/regenerate_embeddings`
- Make sure the Google Generative AI library is installed

#### Empty or Short Responses
- The bot has an automatic retry system for empty responses
- Try adjusting the model temperature settings
- Some models may have token limits that affect response length
- Check if your query is too complex or ambiguous

#### Contextual Retrieval Issues
-   **Check Document Structure**: Poorly structured documents lead to poor context generation.
-   **API Key/Model Access**: Ensure the model used for context generation (e.g., Gemini Flash) is accessible via your OpenRouter key.
-   **Regenerate Embeddings**: If context is consistently poor for a document, try regenerating its embeddings.

### Logging and Debugging

The bot uses a comprehensive logging system:

- Logs are stored in `bot_detailed.log`
- Enable debug mode with `/toggle_debug` for model information
- Check console output for detailed errors
- Image and document operations are logged extensively

### Data Management

If data becomes corrupted:

- Missing embeddings: Re-add documents or use `/regenerate_embeddings`
- Conversation issues: Clear problematic conversations
- Image database problems: Check `images/metadata.json`
- Document storage issues: Verify `documents/` directory integrity
- Empty documents: The bot automatically cleans up empty documents at startup

## Security Considerations

- Bot token should be kept private in the `.env` file
- Consider restricting command access with Discord permissions
- Use admin commands carefully
- Monitor API usage to avoid unexpected costs
- Keep backup copies of important documents
- API keys should never be committed to version control

---

This documentation covers the main aspects of Publicia's functionality and setup. For specific questions or advanced configurations, consult the source code or reach out for additional support.

# Publicia Discord Bot

*A hybrid RAG (Retrieval-Augmented Generation) Discord bot using vector embeddings, BM25 search, and multi-model LLM integration for intelligent document querying and conversational AI.*

*Created for DPS Season 7 as a lore repository for players, Publicia's commands and default system prompt are designed to fit into the setting.*
***A general purpose version without the S7 specific prose and prompts is intended to be made once Season 7 finishes.***

## Overview

Publicia is a Discord bot that functions as an intelligent interface for DPS Season 7's lore documents. By using embeddings to retrieve relevant passages, it augments the context of LLM-generated messages to provide additional context about the setting and lore. At the same time, it maintains a personalised conversation history with each user, roleplaying as an abhuman mentat from the Infinite Empire, the setting of DPS Season 7.

### Key Capabilities
- Answers questions about Ledus Banum 77 and Imperial lore with in-character responses
- Analyzes and processes images with vision-capable AI models
- Remembers conversations with individual users for contextual responses
- Imports documents from text files or Google Docs with automatic refresh
- Answers questions about Ledus Banum 77 and Imperial lore with in-character responses
- Analyzes and processes images with vision-capable AI models
- Remembers conversations with individual users for contextual responses
- Imports documents from text files or Google Docs with automatic refresh and change detection
- Manages documents using robust UUIDs for identification
- Processes `.docx` files to automatically tag specific colored text for lore integration (e.g., `<post-invasion_lore>`)
- Uses multiple AI models with fallback mechanisms for reliability
- Performs advanced hybrid search (semantic + keyword) with contextual retrieval and reranking
- Supports user-selectable AI models for different interaction styles
- Features dynamic temperature control for better response quality
- Manages files within the knowledge base (list, retrieve)

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Discord bot token (from Discord Developer Portal)
- OpenRouter API key (for access to multiple AI models)
- Google API key (for embeddings generation)
- Access to language models via OpenRouter

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/Skelly0/Publicia.git
cd Publicia
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following:
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

# Temperature settings
TEMPERATURE_MIN=0.0
TEMPERATURE_BASE=0.1
TEMPERATURE_MAX=0.4

# Reranking configuration
RERANKING_ENABLED=false # Enable/disable result reranking
RERANKING_CANDIDATES=20 # Number of initial candidates for reranking
RERANKING_MIN_SCORE=0.5 # Minimum relevance score threshold
RERANKING_FILTER_MODE=strict # Options: strict, dynamic, topk

# Contextualization settings
CONTEXTUALIZATION_ENABLED=true # Enable AI-generated context for chunks
MAX_WORDS_FOR_CONTEXT=20000 # Maximum words to consider for context generation
USE_CONTEXTUALISED_CHUNKS=true # Use contextualized chunks in prompts

# Feature toggles
KEYWORD_DATABASE_ENABLED=true # Enable keyword database system
KEYWORD_CHECK_CHUNK_LIMIT=5 # Number of chunks to check for keywords
DOCUMENT_LIST_ENABLED=true # Include internal document list in LLM prompts

# Optional: Auto-process Google Docs with lore tagging
AUTO_PROCESS_GOOGLE_DOCS=false # Set to true to enable automatic .docx download and processing

# Optional: Channel ID for automatic Google Doc tracking
# DOC_TRACKING_CHANNEL_ID=your_channel_id_here

# Optional: Permission settings (comma-separated user/role IDs)
# ALLOWED_USER_IDS=123456789,987654321
# ALLOWED_ROLE_IDS=111111111,222222222
```

4. Run the bot:
```bash
python bot.py
```

## Features

### Image Analysis
Publicia can process, store, and analyze images related to your lore:
- Add images to the knowledge base with automatic or manual descriptions
- Include images in search results when relevant to queries
- Process images attached directly to messages
- View and manage stored images with dedicated commands

### Multiple AI Models
Users can select their preferred AI model for responses:
- **Gemini 2.5 Flash**: Optimized for accuracy, image analysis, and thinking capabilities
- **Qwen QwQ 32B**: Great for roleplaying with strong lore accuracy
- **Qwen 3 235B A22B**: Large parameter model with enhanced capabilities
- **DeepSeek V3 0324**: Advanced conversational model with improved performance
- **DeepSeek-R1**: Best for immersive roleplaying and reasoning
- **Claude 3.5 Haiku**: Fast responses with image capabilities
- **Claude 4 Sonnet**: Premium capabilities (admin restricted)
- **Nous: Hermes 405B**: Balanced between creativity and precision
- **Grok 3 Mini**: X.AI's efficient model for quick responses

Additional admin-only models include Gemini 2.5 Pro, Claude 3.7 Sonnet, Anubis Pro 105B, Llama 4 Maverick, OpenAI GPT-4.1 models, and Phi-4 Multimodal.

The bot includes automatic retry systems when models return blank or extremely short responses, and will fall back to alternative models when needed.

### Advanced Semantic Search
Publicia uses Google's Generative AI embeddings with sophisticated reranking mechanisms:
- Multiple filter modes for improved relevance (strict, dynamic, topk)
- Customizable minimum score threshold
- Weighted combination of initial and reranked scores
- Dynamic parsing limits based on query complexity

### Conversation Memory
The bot remembers interactions with each user, providing continuity across conversations. Each user has their own conversation history that provides context for the AI, and can view or manage their history.

### Google Docs Integration
Automatically fetch and update content from Google Docs to keep your lore fresh without manual updates:
- Track specific Google Docs with custom names
- Periodically refresh content on a schedule
- Extract content from Google Doc links in messages
- Create citations linking back to source documents

### Dynamic Temperature Control
The bot automatically adjusts response temperature based on query type:
- Lower temperature for factual/information queries
- Higher temperature for creative/roleplay scenarios
- Base temperature for balanced queries
- Detects context through analysis of roleplay elements, question markers, etc.
- Customize your personal temperature range with the `/temperature` command. Values must satisfy `0.0 ≤ min ≤ base ≤ max ≤ 2.0`.
### Debugging and Logging
Comprehensive tools to help troubleshoot issues:
- Debug mode is **ON** by default and shows which model generated responses. Use `/toggle_debug` to hide this information
- Export prompts to see exactly what's being sent to the models
- Detailed logs for tracking operations and errors
- Performance monitoring and optimization

## Commands

### Prefix Commands (use with "Publicia!")
| Command | Description |
|---------|-------------|
| `Publicia! add_doc "Document Name"` | Add a document with optional attachment |
| `Publicia! add_image "Image Name" [yes/no]` | Add an image with optional auto-description |
| `Publicia! edit_image [image_id]` | View and edit an image description |

### Slash Commands

#### Query and Conversation
| Command | Description |
|---------|-------------|
| `/query` | Ask a question with optional image URL |
| `/history` | View your conversation history |
| `/manage_history` | View and manage history with indices |
| `/delete_history_messages` | Delete specific messages |
| `/lobotomise` | Wipe your conversation history |
| `/memory_clear` | Wipe your conversation history |
| `/archive_conversation` | Archive your current conversation history |
| `/list_archives` | List your archived conversation histories |
| `/swap_conversation` | Swap between current and archived conversation histories |
| `/delete_archive` | Delete an archived conversation |
| `/parse_channel` | Toggle parsing of channel messages for context |

#### Document Management
| Command | Description |
|---------|-------------|
| `/add_info` | Add text directly to knowledge base (UUID assigned automatically) |
| `/list_docs` | List all documents (shows names and UUIDs) |
| `/remove_doc` | Remove a document by its UUID (`doc_uuid`) |
| `/search_docs` | Search documents directly |
| `/search_keyword` | Find occurrences of a specific keyword |
| `/search_keyword_bm25` | Keyword search ranked with BM25 |
| `/add_googledoc` | Track a Google Doc |
| `/list_googledocs` | List all tracked Google Docs |
| `/remove_googledoc` | Remove a tracked Google Doc |
| `/rename_document` | Rename a document (identified by current UUID or original name) |
| `/retrieve_file` | Retrieve a specific file by its UUID (`doc_uuid`) |
| `/archive_channel` | Archive messages from a Discord channel as a document (admin only). Params: `channel`, `message_limit`, `document_name`, `include_bots`, `include_attachments`. |
| `/summarize_doc` | Generate a summary of a document by its UUID or original name. Param: `identifier`. |
| `/view_chunk` | View the content of a specific document chunk by UUID or original name. Params: `identifier`, `chunk_index`, `contextualized`. |
| `/process_docx_lore` | Process a .docx file to tag specific colored text with XML tags (admin only). Params: `docx_file`, `output_filename`. |
| `/track_channel` | Start tracking a Discord channel and archive it periodically (admin only). Params: `channel`, `update_interval_hours`. |
| `/untrack_channel` | Stop tracking a Discord channel (admin only). Param: `channel`. |

#### Image Management
| Command | Description |
|---------|-------------|
| `/list_images` | List all images in knowledge base |
| `/view_image` | View an image from knowledge base |
| `/remove_image` | Remove an image |
| `/update_image_description` | Update an image description |

#### Settings and Utilities
| Command | Description |
|---------|-------------|
| `/set_model` | Change your preferred AI model |
| `/get_model` | Show your current model |
| `/toggle_debug` | Toggle showing model info in responses |
| `/list_commands` | Show all available commands |
| `/help` | Display detailed help information |
| `/toggle_prompt_mode` | Toggle between standard (immersive) and informational (concise) system prompts |
| `/pronouns` | Set your preferred pronouns |
| `/temperature` | Set your custom temperature range (0.0–2.0). Call with no values to reset to defaults |
| `/whats_new` | Shows documents and images added or updated recently |

#### Admin Commands
| Command | Description |
|---------|-------------|
| `/ban_user` | Ban a user from using the bot |
| `/unban_user` | Unban a user |
| `/reload_docs` | Reload all documents from disk |
| `/regenerate_embeddings` | Regenerate all document embeddings |
| `/refresh_docs` | Manually refresh all tracked Google Docs |
| `/compare_models` | Compare responses from multiple AI models |

## Documentation

For detailed documentation, please see the [Publicia Documentation](documentation/Publicia%20Documentation.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

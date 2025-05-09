# Publicia Discord Bot

*An imperial abhuman mentat interface for Ledus Banum 77 and Imperial lore!*

## Overview

Publicia is a sophisticated Discord bot that serves as an intelligent neural interface to your lore documents. It uses advanced embedding technology to understand and respond to questions about your setting, maintaining personalized conversation history with each user while roleplaying as an abhuman mentat from the Infinite Empire.

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
- Google API key (for embeddings generation and Gemini image features)
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
LLM_MODEL=google/gemini-2.0-flash-001 # Default model for text generation
EMBEDDING_MODEL=models/text-embedding-004 # Model for document embeddings
# GOOGLE_API_KEY is also used for Gemini image generation/editing features
EMBEDDING_DIMENSIONS=1024
TOP_K=10
TOP_K_MULTIPLIER=0.5
API_TIMEOUT=150
MAX_RETRIES=10

# Temperature settings
TEMPERATURE_MIN=0.0
TEMPERATURE_BASE=0.1
TEMPERATURE_MAX=0.4

# Reranking configuration
RERANKING_ENABLED=true
RERANKING_CANDIDATES=20
RERANKING_MIN_SCORE=0.5
RERANKING_FILTER_MODE=strict
```

4. Run the bot:
```bash
python PubliciaV13.py
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
- **DeepSeek-R1**: Best for immersive roleplaying and creative responses
- **Gemini 2.5 Flash**: Best for accuracy, citations, and image analysis
- **Nous: Hermes 405B**: Balanced between creativity and factual precision
- **Qwen QwQ 32B**: Great for roleplaying with strong lore accuracy
- **Claude 3.5 Haiku**: Fast responses with image capabilities
- **Claude 3.5/3.7 Sonnet**: Admin-only premium capabilities
- **Wayfarer 70B**: Optimized for narrative-driven roleplay
- **Anubis Pro 105B**: Enhanced emotional intelligence and prompt adherence

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

### Debugging and Logging
Comprehensive tools to help troubleshoot issues:
- Toggle debug mode to show which model generated responses
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
| `/add_googledoc` | Track a Google Doc |
| `/list_googledocs` | List all tracked Google Docs |
| `/remove_googledoc` | Remove a tracked Google Doc |
| `/rename_document` | Rename a document (identified by current UUID or original name) |
| `/retrieve_file` | Retrieve a specific file by its UUID (`doc_uuid`) |
| `/archive_channel` | Archive messages from a Discord channel as a document (admin only). Params: `channel`, `message_limit`, `document_name`, `include_bots`, `include_attachments`. |
| `/summarize_doc` | Generate a summary of a document by its UUID or original name. Param: `identifier`. |
| `/view_chunk` | View the content of a specific document chunk by UUID or original name. Params: `identifier`, `chunk_index`, `contextualized`. |
| `/process_docx_lore` | Process a .docx file to tag specific colored text with XML tags (admin only). Params: `docx_file`, `output_filename`. |

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
| `/whats_new` | Shows documents and images added or updated recently |

#### Admin Commands
| Command | Description |
|---------|-------------|
| `/ban_user` | Ban a user from using the bot |
| `/unban_user` | Unban a user |
| `/reload_docs` | Reload all documents from disk |
| `/regenerate_embeddings` | Regenerate all document embeddings |
| `/refresh_docs` | Manually refresh all tracked Google Docs |
| `/parse_channel` | Toggle parsing of channel messages for context (Admin can toggle for others) |
| `/compare_models` | Compare responses from multiple AI models |

## Documentation

For detailed documentation, please see the [Publicia Documentation](documentation/Publicia%20Documentation.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

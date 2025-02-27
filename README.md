# Publicia Discord Bot

*An imperial abhuman mentat interface for Ledus Banum 77 and Imperial lore!*

## Overview

Publicia is a sophisticated Discord bot that serves as an intelligent neural interface to your lore documents. It uses advanced embedding technology to understand and respond to questions about your setting, maintaining personalized conversation history with each user while roleplaying as an abhuman mentat from the Infinite Empire.

### Key Capabilities
- Answers questions about Ledus Banum 77 and Imperial lore with in-character responses
- Analyzes and processes images with vision-capable AI models
- Remembers conversations with individual users for contextual responses
- Imports documents from text files or Google Docs with automatic refresh
- Uses multiple AI models with fallback mechanisms for reliability
- Performs semantic search across all knowledge documents and images
- Supports user-selectable AI models for different interaction styles

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Discord bot token (from Discord Developer Portal)
- OpenRouter API key (for access to multiple AI models)
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

# Optional (defaults shown)
LLM_MODEL=deepseek/deepseek-r1
CLASSIFIER_MODEL=google/gemini-2.0-flash-001
TOP_K=10
API_TIMEOUT=150
MAX_RETRIES=10
```

4. Run the bot:
```bash
python PubliciaV8.py
```

## Features

### Image Analysis
Publicia can now process, store, and analyze images related to your lore:
- Add images to the knowledge base with automatic or manual descriptions
- Include images in search results when relevant to queries
- Process images attached directly to messages
- View and manage stored images with dedicated commands

### Multiple AI Models
Users can select their preferred AI model for responses:
- **DeepSeek-R1**: Best for immersive roleplaying and creative responses
- **Gemini 2.0 Flash**: Best for accuracy, citations, and image analysis
- **Nous: Hermes 405B**: Balanced between creativity and factual precision
- **Claude 3.5 Haiku**: Fast responses with image capabilities
- **Claude 3.7 Sonnet**: Admin-only premium capabilities

### Semantic Document Search
Publicia uses embeddings to understand the meaning behind user questions, not just keywords! This allows for more natural queries and better answers with proper citations.

### Conversation Memory
The bot remembers interactions with each user, providing continuity across conversations. Each user has their own conversation history that provides context for the AI, and can view or manage their history.

### Google Docs Integration
Automatically fetch and update content from Google Docs to keep your lore fresh without manual updates:
- Track specific Google Docs with custom names
- Periodically refresh content on a schedule
- Extract content from Google Doc links in messages
- Create citations linking back to source documents

### Debugging and Logging
Comprehensive logging system with debug mode to help troubleshoot issues:
- Toggle debug mode to show which model generated responses
- Detailed logs for tracking operations and errors
- Performance monitoring and optimization

## Commands

### Prefix Commands (use with "Publicia!")
| Command | Description |
|---------|-------------|
| `Publicia! add_doc "Document Name"` | Add a document with optional attachment |
| `Publicia! add_image "Image Name" [yes/no]` | Add an image with optional auto-description |

### Slash Commands

#### Query and Conversation
| Command | Description |
|---------|-------------|
| `/query` | Ask a question with optional image URL |
| `/history` | View your conversation history |
| `/manage_history` | View and manage history with indices |
| `/delete_history_messages` | Delete specific messages |
| `/lobotomise` | Wipe your conversation history |

#### Document Management
| Command | Description |
|---------|-------------|
| `/add_info` | Add text directly to knowledge base |
| `/list_docs` | List all documents |
| `/remove_doc` | Remove a document |
| `/search_docs` | Search documents directly |
| `/add_googledoc` | Track a Google Doc |
| `/list_googledocs` | List all tracked Google Docs |
| `/remove_googledoc` | Remove a tracked Google Doc |
| `/rename_document` | Rename a document |

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

#### Admin Commands
| Command | Description |
|---------|-------------|
| `/ban_user` | Ban a user from using the bot |
| `/unban_user` | Unban a user |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

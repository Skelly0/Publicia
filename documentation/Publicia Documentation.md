# Publicia Bot Documentation

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

Publicia is a sophisticated Discord bot designed to serve as an interactive lore repository for the fictional setting of Ledus Banum 77 and the Infinite Empire. The bot takes on the character of Publicia, an "Imperial abhuman mentat" with enhanced cognitive abilities, allowing users to query information through natural conversation.

### Key Capabilities

- **Document Search & Retrieval**: Uses vector embeddings to find relevant information from documents
- **Image Analysis**: Can process, store, and analyze images related to the lore
- **Conversation Memory**: Remembers conversation history for contextual responses
- **Multiple AI Models**: Supports various AI models with automatic fallback mechanisms
- **Google Doc Integration**: Can fetch and index content from Google Docs
- **Role-Playing**: Maintains character as Publicia while providing information

### Lore Context

The bot is designed to role-play as Publicia, an abhuman mentat specializing in knowledge about Ledus Banum 77 (also known as Tundra) and the Infinite Empire. The character adheres to specific lore elements, including:

- The Infinite Empire: A multiversal entity spanning multiple planes of existence
- Planes: Different layers of reality, accessible via Resonant Drilling
- Ledus Banum 77: A recently conquered planet with valuable Ordinium resources
- Imperial Institutions: Various organizations like House Alpeh, the IMC, etc.

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
│ (Vector Search)│      │ (OpenRouter API)   │
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
2. **DocumentManager**: Manages document storage, embedding generation, and semantic search
3. **ImageManager**: Handles image storage, retrieval, and description generation
4. **ConversationManager**: Maintains conversation history for users
5. **UserPreferencesManager**: Handles user preferences like AI model selection
6. **Config**: Loads and provides configuration settings

### Data Flow

1. User sends a query (message or command)
2. Bot analyzes query with classifier model (usually Gemini)
3. Enhanced search finds relevant documents and images
4. Search results are synthesized into context
5. User's conversation history is added for continuity
6. AI model generates response based on all context
7. Response is sent back to user, potentially with images
8. Conversation history is updated

## Features

### Document Management

The bot uses vector embeddings to store and retrieve documents semantically:

- Documents are chunked into smaller sections for precise retrieval
- Each chunk is converted to vector embeddings using SentenceTransformer
- Queries are matched to the most relevant document chunks
- Similarity scores determine the best matches
- Documents can be added, removed, or searched directly

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
- Implements "LOBOTOMISE" command to wipe history
- JSON-based storage for persistence

### Google Doc Integration

Unique capability to work with Google Docs:

- Track Google Docs with custom names
- Automatically refresh content on schedule
- Extract content from Google Doc links in messages
- Create citations linking back to source documents
- Support renaming and removing tracked documents

### AI Model Selection

Users can choose their preferred AI model:

- DeepSeek-R1: Best for immersive roleplaying
- Gemini 2.0 Flash: Optimized for accuracy and image analysis
- Nous: Hermes 405B: Balanced between creativity and precision
- Claude 3.5 Haiku: Fast responses with image capabilities
- Claude 3.7 Sonnet: Premium capabilities (admin restricted)

Each model has different strengths, fallback mechanisms ensure reliability.

### Search and Retrieval

The search system has multiple stages:

1. **Query Analysis**: Extracts key topics and search terms
2. **Enhanced Search**: Uses analysis to improve search quality
3. **Result Synthesis**: Combines search results into coherent context
4. **Response Generation**: Creates roleplayed response with citations

## Setup Guide

### Prerequisites

- Python 3.8+ with pip
- Discord Bot Token (from Discord Developer Portal)
- OpenRouter API Key
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
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=dgoogle/gemini-2.0-flash-001  # Default model
CLASSIFIER_MODEL=google/gemini-2.0-flash-001  # For query analysis
TOP_K=10  # Number of search results to return
API_TIMEOUT=150  # Seconds
MAX_RETRIES=10
```

5. Create necessary directories
```bash
mkdir -p documents images conversations user_preferences lorebooks
```

### Required Packages

```
discord.py
python-dotenv
sentence-transformers
numpy
aiohttp
apscheduler
torch
```

### Running the Bot

```bash
python PubliciaV8.py
```

The bot will display a startup banner and initialize all components.

## Command Reference

### Slash Commands

#### Document Management

- `/add_info`: Add text directly to knowledge base
  - **Parameters**: `name` (document name), `content` (document content)

- `/list_docs`: Show all documents in knowledge base

- `/remove_doc`: Remove a document
  - **Parameters**: `name` (document to remove)

- `/search_docs`: Search documents directly
  - **Parameters**: `query` (search term)

- `/add_googledoc`: Track a Google Doc
  - **Parameters**: `doc_url` (URL or ID), `name` (optional custom name)

- `/list_googledocs`: List all tracked Google Docs

- `/remove_googledoc`: Remove a tracked Google Doc
  - **Parameters**: `identifier` (ID, URL, or name)

- `/rename_document`: Rename a document
  - **Parameters**: `current_name`, `new_name`

#### Image Management

- `/list_images`: Show all images in knowledge base

- `/view_image`: View an image
  - **Parameters**: `image_id` (ID of image)

- `/remove_image`: Remove an image
  - **Parameters**: `image_id` (ID of image)

- `/update_image_description`: Update image description
  - **Parameters**: `image_id` (ID of image), `description` (new description)

#### Query and Conversation

- `/query`: Ask a question with optional image
  - **Parameters**: `question` (your query), `image_url` (optional)

- `/history`: View your conversation history
  - **Parameters**: `limit` (number of messages, default 10)

- `/manage_history`: View and manage history with indices
  - **Parameters**: `limit` (number of messages)

- `/delete_history_messages`: Delete specific messages
  - **Parameters**: `indices` (comma-separated list), `confirm` (must be "yes")

- `/lobotomise`: Wipe your conversation history

#### Settings and Utilities

- `/set_model`: Change your preferred AI model
  - **Parameters**: `model` (model choice from options)

- `/get_model`: Show your current model

- `/toggle_debug`: Toggle showing model info in responses

- `/list_commands`: Show all available commands

- `/help`: Detailed help information

#### Admin Commands

- `/ban_user`: Ban a user from using the bot
  - **Parameters**: `user` (Discord user)

- `/unban_user`: Unban a user
  - **Parameters**: `user` (Discord user)

### Prefix Commands

These commands use the prefix "Publicia!" instead of slash commands:

- `Publicia! add_doc "Document Name"`: Add document (with optional attachment)

- `Publicia! add_image "Image Name" [yes/no]`: Add image with optional auto-description

## Advanced Usage

### Model Selection Strategy

Choose models based on your needs:

- **DeepSeek-R1**: Use for immersive, creative responses and roleplaying
- **Gemini 2.0 Flash**: Best for factual accuracy, citations, and image analysis
- **Nous: Hermes 405B**: Good middle-ground between creativity and precision
- **Claude 3.5 Haiku**: Fast responses with image capabilities
- **Claude 3.7 Sonnet**: Most advanced capabilities (admin-only)

The bot will automatically fall back to available models if your preferred model fails.

### Document Organization Tips

- Keep document names descriptive but concise
- Group related information in single documents
- Use Google Docs for frequently updated information
- Add citations in documents when possible
- Images with detailed descriptions improve search relevance

### Performance Optimization

- Keep conversation histories manageable (use `/lobotomise` periodically)
- Limit number of tracked Google Docs
- Use specific queries for better search precision
- Consider removing unused documents to reduce search space
- Choose faster models (like Gemini Flash) for simple queries

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

## Troubleshooting

### Common Issues

#### Bot Not Responding
- Ensure Discord bot token is correct
- Check that the bot has proper permissions in Discord
- Verify internet connection and API access

#### Search Not Finding Relevant Results
- Add more detailed documents to knowledge base
- Use more specific queries
- Check document formatting and content
- Adjust TOP_K value in .env

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

#### Image Processing Failures
- Check supported image formats (PNG, JPG, etc.)
- Ensure image URLs are direct links to images
- Try using a different vision-capable model
- Reduce image size if very large

### Logging and Debugging

The bot uses a comprehensive logging system:

- Logs are stored in `bot_detailed.log`
- Enable debug mode with `/toggle_debug` for model information
- Check console output for detailed errors
- Image and document operations are logged extensively

### Data Management

If data becomes corrupted:

- Missing embeddings: Re-add documents
- Conversation issues: Clear problematic conversations
- Image database problems: Check `images/metadata.json`
- Document storage issues: Verify `documents/` directory integrity

## Security Considerations

- Bot token should be kept private in the `.env` file
- Consider restricting command access with Discord permissions
- Use admin commands carefully
- Monitor API usage to avoid unexpected costs
- Keep backup copies of important documents

---

This documentation covers the main aspects of Publicia's functionality and setup. For more specific questions or advanced configurations, consult the source code or reach out for additional support.

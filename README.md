# Publicia Discord Bot

*An imperial abhuman mentat interface for Ledus Banum 77 and Imperial lore!*

## Overview

Publicia is a Discord bot that serves as an intelligent neural interface to your lore documents. It uses advanced embedding technology to understand and respond to questions about your setting, maintaining personalized conversation history with each user.

### Key Capabilities
- Answers questions about Ledus Banum 77 and Imperial lore
- Remembers conversations with individual users
- Imports documents from text files or Google Docs
- Uses AI to provide context-aware responses
- Semantic search across all knowledge documents

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Discord bot token (from Discord Developer Portal)
- OpenRouter API key
- Access to a language model (via OpenRouter)

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/Skelly0/Publicia.git
cd Publicia
```

2. Install required packages
```bash
pip install discord.py python-dotenv sentence-transformers numpy aiohttp apscheduler
```

3. Create a `.env` file in the root directory with the following:
```
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=your_preferred_model
```

4. Create directories for document storage:
```
mkdir -p documents lorebooks conversations
```

5. Run the bot:
```bash
python PubliciaV5.py
```

## Features

### Semantic Document Search
Publicia uses embeddings to understand the meaning behind user questions, not just keywords! This allows for more natural queries and better answers.

### Conversation Memory
The bot remembers interactions with each user, providing continuity across conversations. Each user has their own conversation history that provides context for the AI.

### Google Docs Integration
Automatically fetch and update content from Google Docs to keep your lore fresh without manual updates.

### AI-Powered Responses
Combines conversation history, relevant document context, and advanced AI to give informative and contextually appropriate answers.

### Admin Controls
Moderators can manage users and documents, including banning problematic users if needed.

## Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `/query` | Ask Publicia a question | `/query question:what is the capital of the empire?` |
| `/listcommands` | Show all available commands | `/listcommands` |
| `/lobotomise` | Clear your conversation history | `/lobotomise` |
| `/adddoc` | Add a document | `/adddoc name:empire_history content:long ago...` |
| `/listdocs` | List all documents | `/listdocs` |
| `/add_googledoc` | Track a Google Doc | `/add_googledoc doc_url:https://... name:imperial_navy` |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

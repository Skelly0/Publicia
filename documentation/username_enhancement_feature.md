# Username Enhancement Feature

## Overview

The username enhancement feature automatically improves search query relevance by including the user's nickname/username in queries that are asking about the user themselves or involve roleplay scenarios.

## How It Works

When a user asks a question that appears to be about themselves (e.g., "Do you know me?", "Who am I?", "My character background") or involves roleplay actions (e.g., "I want to go to the castle", "I cast fireball"), the system automatically prepends their username/nickname to the search query before generating embeddings.

### Examples

- **Original query**: "Do you know me?"
- **Enhanced query**: "JohnDoe Do you know me?"

- **Original query**: "I want to go to the castle"
- **Enhanced query**: "JohnDoe I want to go to the castle"

This enhancement helps the embedding search find relevant context about the specific user, making responses more personalized and accurate for both self-referential questions and roleplay scenarios.

## Supported Query Patterns

The system detects self-referential and roleplay queries using the following patterns:

### Direct Self-Reference
- "Do you know me?"
- "Who am I?"
- "What do you know about me?"
- "Tell me about myself"
- "About me"

### Personal Attributes
- "My character"
- "My background"
- "My story"
- "My history"
- Questions starting with "My..."

### Personal Questions
- "Am I...?"
- "Do I...?"
- "Have I...?"
- "Can I...?"
- "Will I...?"
- "Did I...?"

### Memory References
- "Remember me?"
- "Know anything about me?"

### Roleplay Actions
- "I want to..."
- "I try to..."
- "I go to..."
- "I talk to..."
- "I cast..."
- "I use..."
- "I take..."
- "I attack..."
- "I examine..."
- And many other action-oriented patterns

### Roleplay Questions
- "Should I...?"
- "Can I...?"
- "May I...?"
- "Let me..."

### Roleplay Indicators (from Temperature System)
- **Asterisk actions**: `*walks slowly*`, `*casts spell*`, `*examines artifact*`
- **Speech marks**: `I say "Hello"`, `I tell him 'Be careful'`
- **Roleplay phrases**: "says", "speaks to", "performs", "acts", "moves", "appears", "suddenly"
- **Narrative elements**: "scene continues", "begins to", "feeling nervous", "expression"

## Implementation Details

### Functions Added

1. **`_should_enhance_query_with_username(question: str) -> bool`**
   - Determines if a query would benefit from username enhancement
   - Uses regex patterns to detect self-referential language and roleplay scenarios

2. **`_enhance_query_with_username(question: str, username: str) -> str`**
   - Prepends the username to queries that need enhancement
   - Returns the original query unchanged if enhancement isn't needed

### Modified Functions

1. **`process_hybrid_query()`**
   - Now enhances queries before generating embeddings
   - Uses enhanced queries for both context-aware and standard searches
   - Applies enhancement to reranking as well

### Files Modified

- **`bot.py`**: Added enhancement functions and integrated them into the search pipeline
- **`commands/query_commands.py`**: Updated to pass nickname instead of username for better context

## Benefits

1. **Improved Personalization**: Users asking about themselves get more relevant results
2. **Enhanced Roleplay Experience**: Roleplay actions and scenarios get better context matching
3. **Better Context Matching**: Username inclusion helps find user-specific information in documents
4. **Seamless Experience**: Enhancement happens automatically without user intervention
5. **Selective Application**: Only applies to queries that would benefit from it

## Testing

A comprehensive test suite (`test_scripts/test_username_enhancement.py`) verifies:
- Correct detection of self-referential queries
- Proper detection of roleplay scenario queries
- Proper username prepending
- No enhancement of general queries
- Coverage of 47+ different query patterns including roleplay actions

## Usage

The feature works automatically for both:
- Slash commands (`/query`)
- Direct messages to the bot

No configuration or user action is required - the system intelligently determines when to apply username enhancement.

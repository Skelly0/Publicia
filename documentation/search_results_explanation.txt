# Publicia Search Functionality

Publicia uses a sophisticated hybrid search system to provide relevant results to user queries. This document explains the key components and workflow of the search process.

## Overview

The search functionality combines several advanced techniques for optimal relevance:

-   **Hybrid Search:** Merges semantic understanding (vector embeddings) with keyword matching (BM25 algorithm).
-   **Contextual Retrieval:** Enhances document chunks with AI-generated context *before* embedding, significantly improving relevance for ambiguous passages.
-   **Context-Aware Query Processing:** Detects follow-up questions and automatically enhances the query using conversation history, generating a specific context-aware embedding.
-   **Score-Based Fusion:** Intelligently combines scores from embedding and BM25 searches using configurable weights.
-   **Search Result Caching:** Stores results from previous queries to efficiently handle follow-ups like "tell me more".
-   **Re-ranking (Optional):** Applies a sophisticated algorithm to further refine the fused search results based on relevance scores, with a fallback mechanism.

## Workflow

```mermaid
graph LR
    A[User Query] --> B{Context Dependent?};
    B -- Yes --> C(Get Conversation Context);
    C --> D(Enhance Query);
    D --> E(Generate Context-Aware Embedding);
    B -- No --> F(Generate Standard Query Embedding);
    E --> G[Query Embedding Ready];
    F --> G;
    G --> H{Check Cache for Query};
    H -- Cache Hit --> I(Retrieve Cached Results);
    H -- Cache Miss --> J(Perform Hybrid Search);
    J --> K(Vector Search with Embedding);
    J --> L(BM25 Keyword Search);
    K --> M(Combine Results - Score Fusion);
    L --> M;
    M --> N(Cache Results);
    N --> O{Re-ranking Enabled?};
    O -- Yes --> P(Re-rank Fused Results);
    P --> Q[Final Results];
    O -- No --> Q;
    I --> Q;  // Cached results go directly to final
```

## Components and Code References

### 1. Query Processing & Context Handling (`bot.py`)

-   **`DiscordBot.process_hybrid_query`:** Orchestrates the query handling process.
    -   Calls `is_context_dependent_query` to check if the query needs conversation context.
    -   If context-dependent:
        -   Calls `get_conversation_context` to retrieve relevant past messages.
        -   Calls `enhance_context_dependent_query` to rewrite the query incorporating the context.
        -   Calls `generate_context_aware_embedding` to create a specialized embedding for the enhanced query.
    -   If not context-dependent, generates a standard embedding for the raw query.
-   **`DiscordBot.is_context_dependent_query`:** Analyzes the query text and conversation state to determine if it's likely a follow-up.
-   **`DiscordBot.get_conversation_context`:** Extracts key information (like the last user query and bot response) from the history managed by `ConversationManager`.
-   **`DiscordBot.enhance_context_dependent_query`:** Uses an LLM or template to merge the current query with the extracted context into a more complete, standalone query.
-   **`DiscordBot.generate_context_aware_embedding`:** Generates a vector embedding specifically tailored to the enhanced, context-rich query.

### 2. Hybrid Search Execution (`managers/documents.py`)

-   **`DocumentManager.search`:** The main search function orchestrating the hybrid approach.
    -   Takes the generated query embedding (standard or context-aware) as input.
    -   Calls `custom_search_with_embedding` to perform the vector similarity search.
    -   Calls `_search_bm25` to perform the keyword-based BM25 search using the original query text.
    -   Calls `_combine_search_results` to fuse the results from both methods using score normalization and weighting.
    -   If `RERANKING_ENABLED` is true, calls `rerank_results` to apply the final re-ranking step to the fused list.
-   **`DocumentManager.custom_search_with_embedding`:** Executes the vector search against the document embeddings using the provided query embedding.
-   **`DocumentManager._search_bm25`:** Executes the BM25 search against the indexed document text.
-   **`DocumentManager._combine_search_results`:** Implements the score-based fusion logic, normalizing scores from both search types and applying weights (e.g., 60% embedding, 40% BM25) to calculate a final combined score for each result.
-   **`DocumentManager.rerank_results`:** (Optional) Applies a secondary, potentially more computationally intensive, re-ranking model or logic to the top candidates from the fused results, incorporating a fallback if the re-ranking fails.

### 3. Contextual Retrieval & Embeddings (`managers/documents.py`)

-   **`DocumentManager.contextualize_chunks`:** Called during document processing (add/update). Uses an LLM (like Gemini Flash) to generate a concise summary or context for each document chunk based on its content and surrounding chunks.
-   **`DocumentManager.generate_embeddings`:** Generates vector embeddings. Crucially, when processing documents (`task_type='retrieval'`), it prepends the AI-generated context to the chunk *before* generating the embedding. For queries (`task_type='query'` or `'retrieval_query'`), it embeds the raw or context-enhanced query text.
-   **`DocumentManager.get_contextualized_chunk`:** Retrieves the full contextualized chunk text (Context + Original Text) which is then included in the final prompt sent to the main LLM, ensuring the model sees the same enhanced context that was used for embedding and retrieval.

### 4. Caching (`bot.py`)

-   **`DiscordBot.cache_search_results`:** Stores the fused and potentially re-ranked search results (document IDs, scores, content) associated with a specific user query.
-   **`DiscordBot.get_additional_results`:** When a follow-up query is detected and hits the cache, this function retrieves results from the cached list that were *not* shown in the previous turn, allowing the user to see more relevant information without re-running the full search pipeline.

## Summary

Publicia's search process is a multi-stage pipeline designed for high relevance and contextual understanding. It combines semantic and keyword search (hybrid), enhances both documents and queries with AI-generated context, fuses results intelligently, and uses caching and optional re-ranking to deliver accurate, efficient, and context-aware answers.

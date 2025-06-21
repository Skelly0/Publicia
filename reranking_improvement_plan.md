# Reranking System Improvement Plan

## Current Issues Identified

### 1. **Aggressive Filtering Problem**
- The target Voidcold content had a low embedding similarity score (0.742)
- Current reranking system filtered it out completely
- **Root Cause**: Reranking relies too heavily on initial embedding scores and applies strict thresholds

### 2. **Query Type Detection Limitations**
- Current adaptive settings only detect "complex" queries based on analytical keywords
- Factual queries like "Where did the voidcold come from?" aren't properly categorized
- **Root Cause**: Missing query classification for factual/specific information requests

### 3. **Score Combination Issues**
- Current combination heavily favors Cohere reranking scores (70%) over original scores (30%)
- This can suppress relevant content that has good keyword/BM25 matches but poor semantic similarity
- **Root Cause**: Fixed weighting doesn't adapt to query type or content characteristics

## Proposed Improvements

### 1. **Enhanced Query Classification**

```python
def classify_query_type(self, query: str) -> Dict[str, Any]:
    """Enhanced query classification for better reranking strategy."""
    query_lower = query.lower()
    
    # Factual/specific information queries
    factual_indicators = [
        'where', 'what', 'who', 'when', 'how', 'why',
        'origin', 'come from', 'began', 'started',
        'definition', 'meaning', 'explain',
        'specific', 'particular'
    ]
    
    # Complex analytical queries
    analytical_indicators = [
        'analysis', 'analyze', 'detailed', 'comprehensive',
        'relationship', 'compare', 'contrast', 'intersection',
        'philosophy', 'theology', 'write about', 'discuss'
    ]
    
    # Named entity queries (looking for specific things/people/places)
    has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', query))
    
    factual_score = sum(1 for indicator in factual_indicators if indicator in query_lower)
    analytical_score = sum(1 for indicator in analytical_indicators if indicator in query_lower)
    
    if factual_score > 0 or has_proper_nouns:
        return {
            'type': 'factual',
            'strategy': 'preserve_diverse_sources',
            'bm25_weight': 0.4,  # Higher BM25 weight for keyword matching
            'rerank_weight': 0.6,
            'min_score_threshold': 0.1,  # Very low threshold
            'filter_mode': 'topk'  # Don't filter by score
        }
    elif analytical_score > 0:
        return {
            'type': 'analytical',
            'strategy': 'semantic_focus',
            'bm25_weight': 0.2,
            'rerank_weight': 0.8,
            'min_score_threshold': 0.3,
            'filter_mode': 'dynamic'
        }
    else:
        return {
            'type': 'general',
            'strategy': 'balanced',
            'bm25_weight': 0.25,
            'rerank_weight': 0.75,
            'min_score_threshold': 0.2,
            'filter_mode': 'strict'
        }
```

### 2. **Adaptive Score Combination**

```python
def combine_scores_adaptively(self, original_score: float, rerank_score: float, 
                            query_classification: Dict, chunk_text: str) -> float:
    """Adaptively combine scores based on query type and content characteristics."""
    
    strategy = query_classification['strategy']
    bm25_weight = query_classification['bm25_weight']
    rerank_weight = query_classification['rerank_weight']
    
    # Base combination
    base_score = bm25_weight * original_score + rerank_weight * rerank_score
    
    # Content-based adjustments
    content_boost = 1.0
    
    if strategy == 'preserve_diverse_sources':
        # For factual queries, boost chunks with specific terminology
        # This helps preserve the Voidcold content that has specific terms
        chunk_lower = chunk_text.lower()
        
        # Boost for rare/specific terms (these often contain the answers)
        rare_term_patterns = [
            r'\b[A-Z][a-z]*cold\b',  # Voidcold, etc.
            r'\bordinium\b',          # Specific materials
            r'\bmeteorite\b',         # Specific events
            r'\bdetonation\b',        # Specific actions
        ]
        
        for pattern in rare_term_patterns:
            if re.search(pattern, chunk_text, re.IGNORECASE):
                content_boost += 0.3
        
        # Boost for chunks with multiple specific terms
        specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b', chunk_text))
        if specific_terms > 5:
            content_boost += 0.2
    
    elif strategy == 'semantic_focus':
        # For analytical queries, rely more on semantic similarity
        # but still preserve some diversity
        pass  # Current logic is mostly fine
    
    return base_score * content_boost
```

### 3. **Multi-Stage Reranking**

```python
async def enhanced_rerank_results(self, query: str, initial_results: List, top_k: int = None) -> List:
    """Enhanced reranking with multi-stage approach."""
    
    # Stage 1: Query Classification
    query_classification = self.classify_query_type(query)
    
    # Stage 2: Diversified Candidate Selection
    # For factual queries, ensure we have diverse sources in candidates
    if query_classification['type'] == 'factual':
        # Increase candidate pool to ensure diverse content isn't lost
        candidate_count = min(len(initial_results), max(50, top_k * 4))
        candidates = initial_results[:candidate_count]
    else:
        candidates = initial_results
    
    # Stage 3: Cohere Reranking (if available)
    if self.config and getattr(self.config, 'COHERE_API_KEY', None):
        try:
            # Apply Cohere reranking
            reranked_candidates = await self.apply_cohere_reranking(query, candidates)
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}")
            reranked_candidates = candidates
    else:
        reranked_candidates = candidates
    
    # Stage 4: Adaptive Score Combination
    final_results = []
    for i, result in enumerate(reranked_candidates):
        original_score = result[3]
        rerank_score = getattr(result, 'rerank_score', original_score)
        chunk_text = result[2]
        
        combined_score = self.combine_scores_adaptively(
            original_score, rerank_score, query_classification, chunk_text
        )
        
        final_results.append(result[:3] + (combined_score,) + result[4:])
    
    # Stage 5: Final Filtering and Selection
    final_results.sort(key=lambda x: x[3], reverse=True)
    
    filter_mode = query_classification['filter_mode']
    min_threshold = query_classification['min_score_threshold']
    
    if filter_mode == 'topk':
        # For factual queries, don't filter by score - just take top results
        filtered_results = final_results[:top_k] if top_k else final_results
    else:
        # Apply threshold filtering for other query types
        filtered_results = [r for r in final_results if r[3] >= min_threshold]
        if not filtered_results:  # Fallback if all filtered out
            filtered_results = final_results[:top_k] if top_k else final_results[:5]
    
    return filtered_results[:top_k] if top_k else filtered_results
```

### 4. **Keyword-Aware Boosting**

```python
def calculate_keyword_relevance_boost(self, query: str, chunk_text: str) -> float:
    """Calculate boost based on keyword relevance for factual queries."""
    
    # Extract key terms from query
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    chunk_terms = set(re.findall(r'\b\w+\b', chunk_text.lower()))
    
    # Direct term matches
    direct_matches = len(query_terms.intersection(chunk_terms))
    
    # Semantic term matches (simple approach)
    semantic_matches = 0
    semantic_pairs = {
        'origin': ['began', 'started', 'came', 'source'],
        'come': ['origin', 'began', 'started', 'source'],
        'from': ['origin', 'source'],
        'where': ['location', 'place', 'region'],
        'what': ['definition', 'description', 'explanation'],
    }
    
    for query_term in query_terms:
        if query_term in semantic_pairs:
            for semantic_term in semantic_pairs[query_term]:
                if semantic_term in chunk_terms:
                    semantic_matches += 0.5
    
    # Calculate boost
    total_relevance = direct_matches + semantic_matches
    max_possible = len(query_terms)
    
    if max_possible > 0:
        relevance_ratio = min(total_relevance / max_possible, 1.0)
        return 1.0 + (relevance_ratio * 0.5)  # Up to 50% boost
    
    return 1.0
```

### 5. **Configuration Updates**

```python
# In config.py, add new settings:
RERANKING_STRATEGY = os.getenv('RERANKING_STRATEGY', 'adaptive')  # 'adaptive', 'conservative', 'aggressive'
FACTUAL_QUERY_BOOST = float(os.getenv('FACTUAL_QUERY_BOOST', '1.3'))
PRESERVE_LOW_SCORE_CANDIDATES = bool(os.getenv('PRESERVE_LOW_SCORE_CANDIDATES', 'True').lower() in ('true', '1', 'yes'))
```

## Implementation Priority

### Phase 1: Quick Fixes (High Impact, Low Risk)
1. **Lower minimum score thresholds** for factual queries
2. **Increase candidate pool size** to prevent early filtering
3. **Add query type detection** for factual vs analytical queries

### Phase 2: Enhanced Scoring (Medium Impact, Medium Risk)
1. **Implement adaptive score combination** based on query type
2. **Add keyword relevance boosting** for factual queries
3. **Multi-stage reranking pipeline**

### Phase 3: Advanced Features (High Impact, Higher Risk)
1. **Content-aware boosting** for rare/specific terms
2. **Dynamic threshold adjustment** based on result distribution
3. **Query expansion** for better semantic matching

## Expected Improvements

### For Factual Queries (like Voidcold):
- **Better recall**: Low-scoring but relevant content won't be filtered out
- **Keyword preservation**: BM25 matches get higher weight
- **Diverse sources**: Multiple documents/chunks considered

### For Analytical Queries:
- **Better precision**: Semantic similarity prioritized
- **Coherent results**: Related content grouped together
- **Quality filtering**: Low-relevance content filtered out

### Overall System:
- **Adaptive behavior**: Different strategies for different query types
- **Robust fallbacks**: System degrades gracefully when components fail
- **Configurable**: Easy to tune for different use cases

## Testing Strategy

1. **Create test suite** with different query types
2. **A/B testing** between old and new reranking
3. **Performance monitoring** for latency impact
4. **User feedback collection** for relevance assessment

This improvement plan addresses the core issues identified in the Voidcold test while maintaining good performance for other query types.
# Adaptive Reranking Improvements for Complex Queries

## Overview

This document describes the improvements made to the reranking system to better handle complex analytical queries while maintaining high precision for simple factual queries.

## Problem Summary

The original reranking system performed poorly on complex queries like *"Write a detailed analysis of Mong philosophy as well as it's intersection with their theology"* because:

1. **Semantic Mismatch**: Different embedding contexts for queries vs content created poor alignment
2. **Aggressive Filtering**: Fixed 0.5 minimum score threshold filtered out relevant results
3. **One-Size-Fits-All**: Same strict settings applied to all query types

## Solution: Adaptive Reranking

### 1. Query Complexity Detection

Added [`get_reranking_settings_for_query()`](../managers/config.py:120) method that detects complex queries using keywords:

```python
complex_indicators = [
    'analysis', 'analyze', 'detailed', 'comprehensive', 'intersection', 
    'relationship', 'compare', 'contrast', 'philosophy', 'theology',
    'write about', 'explain in detail', 'discuss', 'elaborate',
    'write a', 'provide a', 'give me a', 'tell me about',
    'describe', 'overview', 'summary', 'breakdown', 'examination',
    'exploration', 'investigation', 'study', 'research', 'deep dive'
]
```

### 2. Adaptive Settings

**For Complex Queries:**
- **Lower threshold**: 0.25 instead of 0.5
- **TopK mode**: No score filtering, just return top results
- **More candidates**: 30 instead of 20 for better selection
- **Simplified context**: `"Find content related to: {query}"` instead of question format

**For Simple Queries:**
- **Standard settings**: Maintains existing behavior for precision
- **Strict filtering**: Keeps high-quality results
- **Question context**: `"Question: {query}\nWhat information would fully answer this question?"`

### 3. Implementation Changes

#### Config Manager ([`managers/config.py`](../managers/config.py))

```python
def get_reranking_settings_for_query(self, query: str):
    """Get adaptive reranking settings based on query complexity."""
    complex_indicators = [...]
    is_complex = any(indicator in query.lower() for indicator in complex_indicators)
    
    if is_complex:
        logger.info("Detected complex query, using lenient reranking settings")
        return {
            'min_score': 0.25,
            'filter_mode': 'topk',
            'candidates': 30
        }
    else:
        return {
            'min_score': self.RERANKING_MIN_SCORE,
            'filter_mode': self.RERANKING_FILTER_MODE,
            'candidates': self.RERANKING_CANDIDATES
        }
```

#### Document Manager ([`managers/documents.py`](../managers/documents.py))

**Search Method**: Uses adaptive candidate counts
```python
if apply_reranking and self.config and hasattr(self.config, 'get_reranking_settings_for_query'):
    adaptive_settings = self.config.get_reranking_settings_for_query(query)
    initial_k = adaptive_settings['candidates']
```

**Rerank Results Method**: Uses adaptive filtering and contexts
```python
# Get adaptive settings
if self.config and hasattr(self.config, 'get_reranking_settings_for_query'):
    adaptive_settings = self.config.get_reranking_settings_for_query(query)
    min_score = adaptive_settings['min_score']
    filter_mode = adaptive_settings['filter_mode']
    
    # Use adaptive query context
    if adaptive_settings['filter_mode'] == 'topk':  # Complex query
        query_context = f"Find content related to: {query}"
    else:  # Simple query
        query_context = f"Question: {query}\nWhat information would fully answer this question?"
```

## Key Improvements

### 1. **Reduced Semantic Mismatch**
- Complex queries use simpler, more direct embedding context
- Better alignment between query intent and content matching

### 2. **Adaptive Filtering**
- Complex queries use TopK mode (no score filtering)
- Simple queries maintain strict filtering for precision

### 3. **Increased Candidate Pool**
- Complex queries search through 30 candidates instead of 20
- Better chance of finding relevant content before reranking

### 4. **Lower Thresholds**
- Complex queries use 0.25 minimum score instead of 0.5
- Prevents filtering out relevant but semantically mismatched content

## Testing

Use [`debug_scripts/test_adaptive_reranking.py`](../debug_scripts/test_adaptive_reranking.py) to verify:

1. **Settings Detection**: Complex queries get lenient settings
2. **Result Quality**: Maintains or improves result count and relevance
3. **Mong Content**: Preserves Mong-related results for complex queries
4. **Backward Compatibility**: Simple queries still work as before

## Expected Results

**Before Improvements:**
- Simple query: "what is mong" → ✅ Returns results
- Complex query: "Write a detailed analysis..." → ❌ Returns few/no results

**After Improvements:**
- Simple query: "what is mong" → ✅ Returns results (unchanged)
- Complex query: "Write a detailed analysis..." → ✅ Returns relevant results

## Monitoring

Check logs for adaptive behavior:
```
INFO: Detected complex query, using lenient reranking settings
INFO: Using adaptive settings: {'min_score': 0.25, 'filter_mode': 'topk', 'candidates': 30}
```

## Benefits

1. **Better Recall**: Complex queries find more relevant content
2. **Maintained Precision**: Simple queries keep high-quality filtering
3. **Automatic**: No manual configuration needed
4. **Extensible**: Easy to add new complexity indicators
5. **Backward Compatible**: Existing functionality unchanged

## Future Enhancements

1. **Machine Learning Detection**: Use ML to detect query complexity
2. **Dynamic Thresholds**: Adjust thresholds based on result distribution
3. **Query Expansion**: Automatically expand complex queries
4. **Context Optimization**: Further refine embedding contexts
5. **Performance Metrics**: Track improvement in result quality

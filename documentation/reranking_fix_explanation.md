# Reranking Fix for Complex Queries

## Problem Description

The query system was failing to return results for complex analytical queries like:
> "Write a detailed analysis of Mong philosophy as well as it's intersection with their theology"

While simple queries like "what is mong" worked fine.

## Root Cause Analysis

The issue was in the **reranking system** that filters search results:

### 1. **Aggressive Filtering**
- The reranking system used a fixed minimum score threshold (0.5)
- Complex queries often have semantic mismatches between:
  - Query embedding context: `"Question: {query}\nWhat information would fully answer this question?"`
  - Content embedding context: `"This document contains the following information: {chunk}"`

### 2. **Semantic Mismatch**
- Complex analytical language ("detailed analysis", "intersection") doesn't match well with document content
- Simple direct queries ("what is mong") match much better
- This caused relevant chunks to score below the threshold and get filtered out

### 3. **Fixed Configuration**
- The system used the same strict settings for all queries
- No adaptation based on query complexity

## Solution Implemented

### 1. **Adaptive Reranking Settings**

Added `get_reranking_settings_for_query()` method to [`managers/config.py`](../managers/config.py) that detects complex queries and applies more lenient settings:

```python
def get_reranking_settings_for_query(self, query: str):
    """Get adaptive reranking settings based on query complexity."""
    complex_indicators = [
        'analysis', 'analyze', 'detailed', 'comprehensive', 'intersection', 
        'relationship', 'compare', 'contrast', 'philosophy', 'theology',
        'write about', 'explain in detail', 'discuss', 'elaborate',
        'write a', 'provide a', 'give me a', 'tell me about'
    ]
    
    is_complex = any(indicator in query.lower() for indicator in complex_indicators)
    
    if is_complex:
        return {
            'min_score': 0.25,      # Lower threshold (was 0.5)
            'filter_mode': 'topk',  # No score filtering (was 'strict')
            'candidates': 30        # More candidates (was 20)
        }
    else:
        return {
            'min_score': self.RERANKING_MIN_SCORE,
            'filter_mode': self.RERANKING_FILTER_MODE,
            'candidates': self.RERANKING_CANDIDATES
        }
```

### 2. **Updated Reranking Logic**

Modified [`managers/documents.py`](../managers/documents.py) to use adaptive settings:

- `rerank_results()` method now gets adaptive settings based on query
- `search()` method uses adaptive candidate count
- Filtering logic uses adaptive thresholds

### 3. **Key Changes**

**For Complex Queries:**
- **Lower threshold**: 0.25 instead of 0.5
- **TopK mode**: No score filtering, just return top results
- **More candidates**: 30 instead of 20 for better selection

**For Simple Queries:**
- **Standard settings**: Maintains existing behavior
- **Strict filtering**: Keeps high-quality results

## Testing

### Debug Scripts Created:

1. **`debug_scripts/test_mong_fix.py`** - Comprehensive test of the fix
2. **`debug_scripts/debug_mong_query.py`** - Detailed debugging of the specific issue
3. **`debug_scripts/test_reranking_settings.py`** - General reranking testing

### Expected Results:

**Before Fix:**
- Simple query: "what is mong" → ✅ Returns results
- Complex query: "Write a detailed analysis..." → ❌ Returns no results

**After Fix:**
- Simple query: "what is mong" → ✅ Returns results (unchanged)
- Complex query: "Write a detailed analysis..." → ✅ Returns results (fixed!)

## Usage

The fix is automatic and requires no configuration changes. The system will:

1. **Detect complex queries** automatically based on keywords
2. **Apply lenient settings** for complex analytical queries
3. **Maintain strict settings** for simple factual queries
4. **Log the adaptive behavior** for monitoring

## Monitoring

Check logs for messages like:
```
INFO: Detected complex query, using lenient reranking settings
INFO: Using adaptive settings: {'min_score': 0.25, 'filter_mode': 'topk', 'candidates': 30}
```

## Alternative Solutions Considered

1. **Disable reranking entirely** - Would reduce quality for simple queries
2. **Lower global threshold** - Would reduce precision for all queries
3. **Different embedding contexts** - More complex to implement
4. **Query preprocessing** - Could lose semantic meaning

The adaptive approach provides the best balance of precision and recall.

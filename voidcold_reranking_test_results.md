# Voidcold Search Reranking Test Results

## Test Query
**Question:** "Where did the voidcold come from?"

**Expected Content:** Information about Voidcold's origins, including:
- "The Voidcold's origins are shrouded in myth"
- "One popular theory suggests it began with a large detonation—likely from a meteorite"
- "unleashed particles of an underground Ordinium deposit into the air"
- "This reaction formed a 'shield' blocking all light, creating the Voidcold"

## Test Results Summary

| Metric | Reranking Disabled | Reranking Enabled | Winner |
|--------|-------------------|-------------------|---------|
| Total Results | 15 | 15 | Tie |
| Relevant Chunks | 8 | 4 | **Disabled** |
| Keyword Hits | 19 | 4 | **Disabled** |
| Best Score | 5.577 | 7.188 | Enabled |

## Key Findings

### Reranking Disabled Performance
- ✅ **Found the target content!** - Chunk 15 from "Region 08 - Istrelka" contained the exact expected information
- ✅ **Excellent keyword coverage** - Found 10/11 target keywords in the best chunk
- ✅ **High relevance** - 8 chunks contained relevant keywords
- ✅ **Complete answer** - All key elements present: meteorite, Ordinium deposit, detonation, shield blocking light

### Reranking Enabled Performance
- ❌ **Failed to find target content** - The specific Voidcold information was not in the top 15 results
- ❌ **Poor keyword coverage** - Only found 4 total keyword hits across all results
- ❌ **Low relevance** - Only 4 chunks contained any relevant keywords
- ❌ **Incomplete answer** - Missing critical information about Voidcold origins

## Detailed Analysis

### Target Content Location
The perfect answer was found in **"Region 08 - Istrelka - Chunk 19/19"** with a score of **0.742** when reranking was disabled. This chunk contained:

> "The Voidcold's origins, nature, and potential exploitation by the Republic. that a large detonation (likely from a meteorite) unleashed particles of an underground Ordinium deposit into the air, which somehow reacted in such a way so as to form a 'shield' that blocks all light from entering the affe..."

### Keyword Breakdown
When reranking was **disabled**, the system found:
- `voidcold`: 1 chunk ✅
- `meteorite`: 1 chunk ✅
- `ordinium`: 1 chunk ✅
- `detonation`: 1 chunk ✅
- `underground`: 1 chunk ✅
- `particles`: 2 chunks ✅
- `shield`: 4 chunks ✅
- `light`: 5 chunks ✅
- `origins`: 1 chunk ✅
- `myth`: 1 chunk ✅

When reranking was **enabled**, the system only found:
- `light`: 3 chunks (generic usage)
- `theory`: 1 chunk (unrelated context)

## Conclusion

**For the question "Where did the voidcold come from?", reranking disabled significantly outperforms reranking enabled.**

### Why Reranking Failed Here
1. **Low initial score**: The target content had a relatively low embedding similarity score (0.742)
2. **Reranking threshold**: The reranking system likely filtered out the correct answer due to its low initial score
3. **Context mismatch**: The reranking model may not have recognized the relevance of the Voidcold content to the query

### Recommendation
For factual queries seeking specific information that might be buried in lower-scoring chunks, **disabling reranking** provides better recall and ensures important content isn't filtered out prematurely.

## Test Configuration
- **Documents loaded**: 38
- **Search results requested**: 15 (top_k)
- **Reranking candidates**: 50
- **Reranking min score**: 0.2
- **Contextualised chunks**: Enabled
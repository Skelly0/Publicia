# Reranking Comparison Test Results

## Test Overview
This test compared the search performance for finding the leader of Krstovlje (expected answer: Abstrata) with reranking enabled vs disabled.

## Configuration
- **Documents loaded**: 38
- **USE_CONTEXTUALISED_CHUNKS**: True (both tests)
- **Test queries**: 
  - "leader of Krstovlje"
  - "Krstovlje leader" 
  - "who leads Krstovlje"
  - "Krstovlje Abstrata"
  - "Abstrata"
  - "Krstovlje"

## Key Findings

### Reranking Disabled (Baseline)
✅ **SUCCESSFUL** - Found relevant chunks containing both 'Krstovlje' and 'Abstrata' for ALL queries

**Performance highlights:**
- Query "leader of Krstovlje": Found relevant chunk at position #10 (score: 0.612)
- Query "Krstovlje leader": Found relevant chunk at position #10 (score: 0.591) 
- Query "who leads Krstovlje": Found relevant chunk at position #10 (score: 0.592)
- Query "Krstovlje Abstrata": Found multiple relevant chunks in top 10
- Query "Abstrata": Found multiple relevant chunks in top 10
- Query "Krstovlje": Found multiple relevant chunks in top 10

### Reranking Enabled 
❌ **DEGRADED PERFORMANCE** - Failed to find relevant chunks for leadership queries

**Performance issues:**
- Query "leader of Krstovlje": **FAILED** - Only found 'Krstovlje' but not 'Abstrata'
- Query "Krstovlje leader": **FAILED** - Only found 'Krstovlje' but not 'Abstrata'  
- Query "who leads Krstovlje": **FAILED** - Only found 'Krstovlje' but not 'Abstrata'
- Query "Krstovlje Abstrata": ✅ Found relevant chunks (when both terms explicitly mentioned)
- Query "Abstrata": ✅ Found relevant chunks
- Query "Krstovlje": ✅ Found relevant chunks

**Technical Issues:**
- Multiple Cohere rerank API errors: `model 'rerank-3.5' not found`
- Reranking system appears to be filtering out the most relevant results for leadership queries

## Detailed Analysis

### Critical Chunk Missing from Reranked Results
The most important chunk for answering "leader of Krstovlje" queries is:
- **Region 07 - Zdrolan - Chunk 3/39** (score: 0.612)
- Contains: "Krstovlje's resistance and the rise of the Zdrolan Independence Front (ZIF)...Abstrata"

**Without reranking**: This chunk appears at position #10 for leadership queries
**With reranking**: This chunk is completely filtered out from the top 10 results

### Reranking System Issues

1. **API Configuration Problem**: 
   - Error: `model 'rerank-3.5' not found`
   - Suggests the reranking model name or API access is misconfigured

2. **Over-aggressive Filtering**:
   - When reranking works, it appears to filter out lower-scored but highly relevant chunks
   - The relevant chunk (score: 0.612) is being excluded in favor of higher-scored but less relevant chunks

3. **Query Type Sensitivity**:
   - Reranking works better for explicit term matching ("Krstovlje Abstrata")
   - Fails for semantic leadership queries ("leader of Krstovlje", "who leads Krstovlje")

## Recommendations

### Immediate Actions
1. **Fix Cohere API Configuration**:
   - Verify the reranking model name (should be 'rerank-english-v3.0' or similar)
   - Check API key permissions for the rerank model

2. **Adjust Reranking Parameters**:
   - Increase `RERANKING_CANDIDATES` from 30 to 50+ to capture more initial results
   - Lower `RERANKING_MIN_SCORE` from 0.45 to 0.3 or lower
   - Consider changing `RERANKING_FILTER_MODE` from 'strict' to 'dynamic' or 'topk'

### Configuration Suggestions
```env
RERANKING_ENABLED=True
RERANKING_CANDIDATES=50
RERANKING_MIN_SCORE=0.3
RERANKING_FILTER_MODE=topk
```

## Conclusion

**Current recommendation: Keep reranking DISABLED** until the following issues are resolved:

1. ❌ Cohere API model configuration errors
2. ❌ Over-aggressive filtering removing relevant results
3. ❌ Poor performance on semantic leadership queries

The baseline search (without reranking) successfully finds the correct answer (Abstrata as leader of Krstovlje) for all test queries, while reranking fails on the most important leadership-specific queries.

**Performance Summary:**
- **Reranking Disabled**: 6/6 queries successful ✅
- **Reranking Enabled**: 3/6 queries successful ❌

The reranking system needs significant tuning before it provides value over the baseline search performance.
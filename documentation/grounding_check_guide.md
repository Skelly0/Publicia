# Grounding Check with RAG - User Guide

## Overview

The Grounding Check system in Publicia implements functionality similar to Google Cloud Discovery Engine's grounding check API. It analyzes how well an answer candidate (generated text) is supported by a given set of facts (reference documents), providing scores and citations to help evaluate the reliability of AI-generated responses.

## Key Concepts

### What is Grounding?

**Grounding** refers to how well a piece of text is supported by factual evidence. A well-grounded response means that every claim in the answer can be traced back to and is supported by the provided facts.

### Perfect Grounding Requirements

- Every claim in the answer candidate must be supported by one or more facts
- Claims must be **wholly entailed** by the facts (not just partially correct)
- Example: "Google was founded by Larry Page and Sergey Brin in 1975" would be considered ungrounded because while the founders are correct, the date is wrong

### Components

1. **Answer Candidate**: The text you want to check for grounding (max 4096 tokens)
2. **Facts**: Reference texts used to verify claims (up to 200 facts, max 10k characters each)
3. **Claims**: Individual sentences extracted from the answer candidate
4. **Citations**: References to facts that support each claim
5. **Support Score**: Overall score (0-1) indicating how well the answer is grounded

## Available Commands

### `/test_grounding`
**Description**: Test the grounding system with built-in Titanic movie examples
**Usage**: Simply run the command to see how different answer candidates score against known facts

### `/grounding_example`
**Description**: Display comprehensive examples and usage instructions
**Usage**: Run to see detailed examples of all grounding check features

### `/check_grounding`
**Description**: Perform a manual grounding check with custom facts
**Parameters**:
- `answer_candidate` (required): The text to check for grounding
- `facts` (required): JSON array of facts in the format below
- `citation_threshold` (optional): Minimum similarity for citations (0.0-1.0, default: 0.6)
- `enable_claim_scores` (optional): Show individual claim scores (default: false)
- `enable_contradictions` (optional): Detect contradicting information (default: false)
- `enable_helpfulness` (optional): Measure how well the answer addresses the prompt (default: false)
- `prompt` (optional): Original question for helpfulness scoring

**Facts JSON Format**:
```json
[
  {
    "fact_text": "Your factual text here",
    "attributes": {
      "author": "Source name",
      "title": "Document title",
      "date": "2024"
    }
  }
]
```

### `/check_grounding_simple`
**Description**: Check grounding using Publicia's document database
**Parameters**:
- `answer_candidate` (required): The text to check for grounding
- `search_query` (required): Query to find relevant documents
- `citation_threshold` (optional): Minimum similarity for citations (default: 0.6)
- `max_facts` (optional): Maximum documents to retrieve (1-20, default: 5)

## Understanding the Results

### Support Score (0.0 - 1.0)
- **0.9-1.0**: Excellent grounding - nearly all claims are well-supported
- **0.7-0.8**: Good grounding - most claims are supported
- **0.5-0.6**: Fair grounding - some claims lack support
- **0.0-0.4**: Poor grounding - many unsupported claims

### Contradiction Score (0.0 - 1.0)
- **0.0-0.2**: Low contradiction - minimal conflicting information
- **0.3-0.5**: Moderate contradiction - some conflicting claims
- **0.6-1.0**: High contradiction - significant conflicts with facts

### Helpfulness Score (0.0 - 1.0)
- **0.8-1.0**: Very helpful - directly answers the question
- **0.6-0.7**: Helpful - addresses most aspects of the question
- **0.4-0.5**: Somewhat helpful - partial answer
- **0.0-0.3**: Not helpful - doesn't address the question

### Citations
- **Numbers in brackets**: `[0]`, `[1]` refer to fact indices
- **Empty citations**: `[]` means no supporting facts found
- **Anti-citations**: Facts that contradict the claim

## Example Usage Scenarios

### 1. Fact-Checking AI Responses
```
/check_grounding_simple
answer_candidate: "The Empire was founded in 2387 by Emperor Marcus."
search_query: "Empire founding date Marcus emperor"
```

### 2. Verifying Historical Claims
```
/check_grounding
answer_candidate: "World War II ended in 1945."
facts: [{"fact_text": "World War II ended on September 2, 1945, when Japan formally surrendered.", "attributes": {"source": "History Textbook"}}]
```

### 3. Checking Creative Writing Accuracy
```
/check_grounding_simple
answer_candidate: "The starship Enterprise was commanded by Captain Kirk."
search_query: "Enterprise starship captain Kirk Star Trek"
enable_claim_scores: true
```

## Best Practices

### For Facts
1. **Break down large texts**: Instead of one massive fact, use multiple smaller, focused facts
2. **Include metadata**: Add author, source, date information in attributes
3. **Use specific facts**: More specific facts lead to better grounding detection
4. **Avoid redundancy**: Don't repeat the same information across multiple facts

### For Citation Thresholds
- **0.8-0.9**: Very strict - only strong matches count as citations
- **0.6-0.7**: Balanced - good for most use cases
- **0.4-0.5**: Lenient - more citations but potentially weaker matches
- **0.2-0.3**: Very lenient - may include tangentially related facts

### For Answer Candidates
- **Keep it focused**: Shorter, more focused answers are easier to ground
- **Avoid speculation**: Stick to factual claims that can be verified
- **Be specific**: Specific claims are easier to ground than vague statements

## Technical Details

### Claim Extraction
The system automatically splits answer candidates into individual claims (typically sentences). Some sentences may not require grounding checks:
- Conversational phrases: "Here is what I found"
- Greetings: "Hello", "Thank you"
- Very short phrases: Less than 3 words

### Similarity Computation
The system uses semantic similarity based on sentence embeddings to determine how well claims match facts. This goes beyond simple keyword matching to understand meaning and context.

### Model Information
- **Default Model**: `all-MiniLM-L6-v2` (lightweight, fast)
- **Fallback**: Automatic fallback to smaller models if needed
- **Performance**: Typically <500ms for most grounding checks

## Troubleshooting

### Common Issues

**"No citations found"**
- Lower the citation threshold
- Check if facts are relevant to the claims
- Ensure facts contain the information being claimed

**"Low support score despite correct information"**
- Facts might be too general or too specific
- Try rephrasing facts to match claim language
- Check for typos or formatting issues

**"High contradiction score unexpectedly"**
- Review facts for negation words (not, never, without)
- Check for conflicting information in fact set
- Ensure facts are consistent with each other

**"Command timeout"**
- Large fact sets or long answer candidates take more time
- Try reducing the number of facts or splitting long answers
- Check internet connection for model downloads

### Error Messages

- **"Invalid JSON format"**: Check facts JSON syntax
- **"Citation threshold must be between 0.0 and 1.0"**: Adjust threshold parameter
- **"At least one fact must be provided"**: Include at least one fact in the array
- **"Max facts must be between 1 and 20"**: Adjust max_facts parameter

## Integration with Publicia

The grounding check system integrates seamlessly with Publicia's existing document management:

1. **Document Search Integration**: Use `/check_grounding_simple` to automatically find relevant facts from your document database
2. **Conversation Context**: Check how well AI responses are grounded in your lore documents
3. **Quality Assurance**: Verify that bot responses are well-supported by your knowledge base

## Advanced Features

### Claim-Level Scoring
When enabled, shows individual support scores for each claim, helping identify which specific parts of an answer are well-grounded.

### Contradiction Detection
Identifies when claims directly contradict the provided facts, useful for detecting hallucinations or errors.

### Helpfulness Scoring
Measures how well the answer addresses the original question, combining grounding with relevance.

## API-Style Usage

For programmatic access, the grounding system can be used directly:

```python
from managers.grounding import GroundingManager, GroundingFact, GroundingSpec

# Initialize manager
grounding_manager = GroundingManager()

# Create facts
facts = [GroundingFact(fact_text="...", attributes={"source": "..."})]

# Configure grounding
spec = GroundingSpec(citation_threshold=0.6, enable_claim_level_score=True)

# Check grounding
response = await grounding_manager.check_grounding(
    answer_candidate="...",
    facts=facts,
    grounding_spec=spec
)
```

This enables integration with custom workflows and automated quality checking systems.

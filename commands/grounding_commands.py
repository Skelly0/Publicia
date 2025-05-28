"""
Grounding check commands for Publicia
"""
import discord
from discord import app_commands
from discord.ext import commands
import logging
import json
import asyncio
from typing import List, Optional
from utils.helpers import split_message, check_permissions
from managers.grounding import GroundingManager, GroundingFact, GroundingSpec

logger = logging.getLogger(__name__)

def register_commands(bot):
    """Register all grounding check commands with the bot."""
    
    # Initialize grounding manager
    grounding_manager = GroundingManager()
    
    @bot.tree.command(name="check_grounding", description="Check how well an answer is grounded in provided facts")
    @app_commands.describe(
        answer_candidate="The answer/text to check for grounding",
        facts="JSON array of facts to check against (format: [{\"fact_text\": \"...\", \"attributes\": {\"author\": \"...\"}}])",
        citation_threshold="Minimum similarity threshold for citations (0.0-1.0, default: 0.6)",
        enable_claim_scores="Enable individual claim-level scores (default: false)",
        enable_contradictions="Enable contradiction detection (default: false)",
        enable_helpfulness="Enable helpfulness scoring (default: false)",
        prompt="Original question/prompt for helpfulness scoring (optional)"
    )
    async def check_grounding(
        interaction: discord.Interaction, 
        answer_candidate: str,
        facts: str,
        citation_threshold: float = 0.6,
        enable_claim_scores: bool = False,
        enable_contradictions: bool = False,
        enable_helpfulness: bool = False,
        prompt: str = None
    ):
        try:
            await interaction.response.defer()
            
            # Validate citation threshold
            if not 0.0 <= citation_threshold <= 1.0:
                await interaction.followup.send("❌ Citation threshold must be between 0.0 and 1.0")
                return
            
            # Parse facts JSON
            try:
                facts_data = json.loads(facts)
                if not isinstance(facts_data, list):
                    raise ValueError("Facts must be a JSON array")
                
                grounding_facts = []
                for fact_data in facts_data:
                    if not isinstance(fact_data, dict) or "fact_text" not in fact_data:
                        raise ValueError("Each fact must have a 'fact_text' field")
                    
                    attributes = fact_data.get("attributes", {})
                    grounding_facts.append(GroundingFact(
                        fact_text=fact_data["fact_text"],
                        attributes=attributes
                    ))
                
            except json.JSONDecodeError as e:
                await interaction.followup.send(f"❌ Invalid JSON format for facts: {e}")
                return
            except ValueError as e:
                await interaction.followup.send(f"❌ Invalid facts format: {e}")
                return
            
            if not grounding_facts:
                await interaction.followup.send("❌ At least one fact must be provided")
                return
            
            # Create grounding spec
            grounding_spec = GroundingSpec(
                citation_threshold=citation_threshold,
                enable_claim_level_score=enable_claim_scores,
                enable_anti_citations=enable_contradictions,
                enable_helpfulness_score=enable_helpfulness
            )
            
            # Perform grounding check
            try:
                response = await grounding_manager.check_grounding(
                    answer_candidate=answer_candidate,
                    facts=grounding_facts,
                    grounding_spec=grounding_spec,
                    prompt=prompt
                )
                
                # Format response
                formatted_response = grounding_manager.format_grounding_response(response, include_details=True)
                
                # Split message if too long
                messages = split_message(formatted_response, max_length=2000)
                
                for i, message in enumerate(messages):
                    if i == 0:
                        await interaction.followup.send(message)
                    else:
                        await interaction.followup.send(message)
                
            except Exception as e:
                logger.error(f"Error during grounding check: {e}")
                await interaction.followup.send(f"❌ Error performing grounding check: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in check_grounding command: {e}")
            try:
                await interaction.followup.send(f"❌ An error occurred: {str(e)}")
            except:
                pass
    
    @bot.tree.command(name="check_grounding_simple", description="Simple grounding check using document search")
    @app_commands.describe(
        answer_candidate="The answer/text to check for grounding",
        search_query="Query to search documents for relevant facts",
        citation_threshold="Minimum similarity threshold for citations (0.0-1.0, default: 0.6)",
        max_facts="Maximum number of facts to retrieve from search (default: 5)"
    )
    async def check_grounding_simple(
        interaction: discord.Interaction,
        answer_candidate: str,
        search_query: str,
        citation_threshold: float = 0.6,
        max_facts: int = 5
    ):
        try:
            await interaction.response.defer()
            
            # Validate parameters
            if not 0.0 <= citation_threshold <= 1.0:
                await interaction.followup.send("❌ Citation threshold must be between 0.0 and 1.0")
                return
            
            if max_facts < 1 or max_facts > 20:
                await interaction.followup.send("❌ Max facts must be between 1 and 20")
                return
            
            # Get document manager from bot
            if not hasattr(bot, 'document_manager'):
                await interaction.followup.send("❌ Document manager not available")
                return
            
            # Search for relevant documents
            try:
                search_results = await bot.document_manager.search_documents(
                    query=search_query,
                    top_k=max_facts,
                    user_id=str(interaction.user.id)
                )
                
                if not search_results:
                    await interaction.followup.send("❌ No relevant documents found for the search query")
                    return
                
                # Convert search results to grounding facts
                grounding_facts = []
                for result in search_results[:max_facts]:
                    # Extract document info
                    doc_name = result.get('document_name', 'Unknown Document')
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    
                    grounding_facts.append(GroundingFact(
                        fact_text=content,
                        attributes={
                            "document_name": doc_name,
                            "search_score": str(round(score, 3))
                        }
                    ))
                
                # Create grounding spec
                grounding_spec = GroundingSpec(
                    citation_threshold=citation_threshold,
                    enable_claim_level_score=True,
                    enable_anti_citations=False,
                    enable_helpfulness_score=True
                )
                
                # Perform grounding check
                response = await grounding_manager.check_grounding(
                    answer_candidate=answer_candidate,
                    facts=grounding_facts,
                    grounding_spec=grounding_spec,
                    prompt=search_query
                )
                
                # Format response with additional context
                formatted_response = f"**Grounding Check Against Document Search**\n"
                formatted_response += f"Search Query: \"{search_query}\"\n"
                formatted_response += f"Documents Found: {len(grounding_facts)}\n\n"
                formatted_response += grounding_manager.format_grounding_response(response, include_details=True)
                
                # Split message if too long
                messages = split_message(formatted_response, max_length=2000)
                
                for i, message in enumerate(messages):
                    if i == 0:
                        await interaction.followup.send(message)
                    else:
                        await interaction.followup.send(message)
                
            except Exception as e:
                logger.error(f"Error during document search: {e}")
                await interaction.followup.send(f"❌ Error searching documents: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in check_grounding_simple command: {e}")
            try:
                await interaction.followup.send(f"❌ An error occurred: {str(e)}")
            except:
                pass
    
    @bot.tree.command(name="grounding_example", description="Show example usage of grounding check functionality")
    async def grounding_example(interaction: discord.Interaction):
        try:
            await interaction.response.defer()
            
            example_text = """
**Grounding Check Examples**

**1. Basic Grounding Check:**
```
/check_grounding
answer_candidate: "Titanic was directed by James Cameron. It was released in 1997."
facts: [{"fact_text": "Titanic is a 1997 American epic romantic disaster movie. It was directed, written, and co-produced by James Cameron.", "attributes": {"author": "Wikipedia"}}]
```

**2. Simple Document-Based Check:**
```
/check_grounding_simple
answer_candidate: "The Empire was founded in 2387."
search_query: "Empire founding date history"
```

**3. Advanced Grounding Check with All Features:**
```
/check_grounding
answer_candidate: "The movie won 11 Academy Awards."
facts: [{"fact_text": "The movie won 11 Academy Awards, and was nominated for fourteen total Academy Awards.", "attributes": {"source": "IMDb"}}]
citation_threshold: 0.7
enable_claim_scores: true
enable_contradictions: true
enable_helpfulness: true
prompt: "How many awards did the movie win?"
```

**Facts JSON Format:**
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

**Key Parameters:**
- `citation_threshold`: Higher values (0.7-0.9) = stricter citations, fewer but stronger matches
- `enable_claim_scores`: Shows individual claim support scores
- `enable_contradictions`: Detects contradicting information
- `enable_helpfulness`: Measures how well the answer addresses the prompt

**Interpretation:**
- **Support Score**: 0.0-1.0, higher = better grounded in facts
- **Citations**: Which facts support each claim
- **Contradiction Score**: 0.0-1.0, higher = more contradictions found
- **Helpfulness Score**: 0.0-1.0, higher = better answers the question
"""
            
            await interaction.followup.send(example_text)
            
        except Exception as e:
            logger.error(f"Error in grounding_example command: {e}")
            try:
                await interaction.followup.send(f"❌ An error occurred: {str(e)}")
            except:
                pass
    
    @bot.tree.command(name="test_grounding", description="Test grounding check with built-in Titanic example")
    async def test_grounding(interaction: discord.Interaction):
        try:
            await interaction.response.defer()
            
            # Built-in example facts
            facts_data = [
                {
                    "fact_text": "Titanic is a 1997 American epic romantic disaster movie. It was directed, written, and co-produced by James Cameron. The movie is about the 1912 sinking of the RMS Titanic. It stars Kate Winslet and Leonardo DiCaprio. The movie was released on December 19, 1997. It received positive critical reviews. The movie won 11 Academy Awards, and was nominated for fourteen total Academy Awards.",
                    "attributes": {"author": "Simple Wikipedia"}
                },
                {
                    "fact_text": "James Cameron's \"Titanic\" is an epic, action-packed romance set against the ill-fated maiden voyage of the R.M.S. Titanic; the pride and joy of the White Star Line and, at the time, the largest moving object ever built. She was the most luxurious liner of her era -- the \"ship of dreams\" -- which ultimately carried over 1,500 people to their death in the ice cold waters of the North Atlantic in the early hours of April 15, 1912.",
                    "attributes": {"author": "Rotten Tomatoes"}
                }
            ]
            
            grounding_facts = [
                GroundingFact(fact_text=fact["fact_text"], attributes=fact["attributes"])
                for fact in facts_data
            ]
            
            # Test different answer candidates
            test_cases = [
                {
                    "answer": "Titanic was directed by James Cameron. It was released in 1997.",
                    "description": "Well-grounded answer"
                },
                {
                    "answer": "Titanic was directed by James Cameron. It starred Brad Pitt and Kate Winslet.",
                    "description": "Partially incorrect answer (wrong actor)"
                },
                {
                    "answer": "Here is what I found. Titanic was directed by James Cameron.",
                    "description": "Answer with non-factual preamble"
                }
            ]
            
            grounding_spec = GroundingSpec(
                citation_threshold=0.6,
                enable_claim_level_score=True,
                enable_anti_citations=True,
                enable_helpfulness_score=True
            )
            
            results = []
            results.append("**Grounding Check Test Results**\n")
            
            for i, test_case in enumerate(test_cases, 1):
                results.append(f"**Test Case {i}: {test_case['description']}**")
                results.append(f"Answer: \"{test_case['answer']}\"")
                
                try:
                    response = await grounding_manager.check_grounding(
                        answer_candidate=test_case['answer'],
                        facts=grounding_facts,
                        grounding_spec=grounding_spec,
                        prompt="Who directed Titanic and when was it released?"
                    )
                    
                    results.append(f"Support Score: {response.support_score:.2f}")
                    if response.contradiction_score > 0:
                        results.append(f"Contradiction Score: {response.contradiction_score:.2f}")
                    if response.helpfulness_score > 0:
                        results.append(f"Helpfulness Score: {response.helpfulness_score:.2f}")
                    
                    # Show claim details
                    for j, claim in enumerate(response.claims):
                        if claim.grounding_check_required:
                            citation_info = f"Citations: {claim.citations}" if claim.citations else "No citations"
                            results.append(f"  Claim {j}: \"{claim.text}\" - {citation_info}")
                    
                    results.append("")  # Empty line between test cases
                    
                except Exception as e:
                    results.append(f"Error: {str(e)}")
                    results.append("")
            
            # Join results and split if necessary
            full_response = "\n".join(results)
            messages = split_message(full_response, max_length=2000)
            
            for i, message in enumerate(messages):
                if i == 0:
                    await interaction.followup.send(message)
                else:
                    await interaction.followup.send(message)
            
        except Exception as e:
            logger.error(f"Error in test_grounding command: {e}")
            try:
                await interaction.followup.send(f"❌ An error occurred: {str(e)}")
            except:
                pass

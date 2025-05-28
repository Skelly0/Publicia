"""
Grounding check functionality for RAG (Retrieval Augmented Generation)
Based on Google Cloud Discovery Engine grounding check API concepts
"""
import re
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


@dataclass
class GroundingFact:
    """Represents a fact used for grounding checks"""
    fact_text: str
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class GroundingClaim:
    """Represents a claim extracted from an answer candidate"""
    text: str
    start_pos: int
    end_pos: int
    support_score: float = 0.0
    citations: List[int] = None
    anti_citations: List[int] = None
    grounding_check_required: bool = True
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.anti_citations is None:
            self.anti_citations = []


@dataclass
class CitedChunk:
    """Represents a chunk of text that supports claims"""
    fact_index: int
    chunk_text: str
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class GroundingSpec:
    """Configuration for grounding checks"""
    citation_threshold: float = 0.6
    anti_citation_threshold: float = 0.8
    enable_claim_level_score: bool = False
    enable_anti_citations: bool = False
    enable_helpfulness_score: bool = False


@dataclass
class GroundingResponse:
    """Response from grounding check"""
    support_score: float
    contradiction_score: float = 0.0
    helpfulness_score: float = 0.0
    claims: List[GroundingClaim] = None
    cited_chunks: List[CitedChunk] = None
    
    def __post_init__(self):
        if self.claims is None:
            self.claims = []
        if self.cited_chunks is None:
            self.cited_chunks = []


class GroundingManager:
    """
    Manager for checking grounding of answer candidates against facts
    Implements functionality similar to Google Cloud Discovery Engine grounding check API
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the grounding manager
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded grounding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load grounding model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded fallback grounding model: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def _extract_claims(self, answer_candidate: str) -> List[GroundingClaim]:
        """
        Extract claims (sentences) from the answer candidate
        
        Args:
            answer_candidate: The text to extract claims from
            
        Returns:
            List of GroundingClaim objects
        """
        # Split into sentences using regex
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, answer_candidate.strip())
        
        claims = []
        current_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Find the actual position in the original text
            start_pos = answer_candidate.find(sentence, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(sentence)
            current_pos = end_pos
            
            # Determine if grounding check is required
            grounding_required = self._requires_grounding_check(sentence)
            
            claim = GroundingClaim(
                text=sentence,
                start_pos=start_pos,
                end_pos=end_pos,
                grounding_check_required=grounding_required
            )
            claims.append(claim)
        
        return claims
    
    def _requires_grounding_check(self, sentence: str) -> bool:
        """
        Determine if a sentence requires grounding check
        
        Args:
            sentence: The sentence to check
            
        Returns:
            True if grounding check is required, False otherwise
        """
        # Patterns that typically don't require grounding
        non_factual_patterns = [
            r'^(here is|here are|i found|let me|according to)',
            r'^(please|thank you|you\'re welcome)',
            r'^(hello|hi|goodbye|bye)',
            r'^\s*$',  # Empty or whitespace only
        ]
        
        sentence_lower = sentence.lower().strip()
        
        for pattern in non_factual_patterns:
            if re.match(pattern, sentence_lower):
                return False
        
        # If sentence is very short and doesn't contain factual content
        if len(sentence.split()) < 3:
            return False
            
        return True
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def _find_supporting_facts(self, claim: str, facts: List[GroundingFact], 
                              threshold: float = 0.6) -> List[int]:
        """
        Find facts that support a given claim
        
        Args:
            claim: The claim to find support for
            facts: List of facts to search through
            threshold: Minimum similarity threshold for support
            
        Returns:
            List of fact indices that support the claim
        """
        supporting_indices = []
        
        for i, fact in enumerate(facts):
            similarity = self._compute_semantic_similarity(claim, fact.fact_text)
            if similarity >= threshold:
                supporting_indices.append(i)
        
        return supporting_indices
    
    def _find_contradicting_facts(self, claim: str, facts: List[GroundingFact],
                                 threshold: float = 0.8) -> List[int]:
        """
        Find facts that contradict a given claim
        
        Args:
            claim: The claim to find contradictions for
            facts: List of facts to search through
            threshold: Minimum similarity threshold for contradiction
            
        Returns:
            List of fact indices that contradict the claim
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated contradiction detection
        contradicting_indices = []
        
        # Look for explicit contradictions using negation patterns
        negation_patterns = [
            r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bwithout\b',
            r'\bincorrect\b', r'\bfalse\b', r'\bwrong\b'
        ]
        
        claim_lower = claim.lower()
        
        for i, fact in enumerate(facts):
            fact_lower = fact.fact_text.lower()
            
            # Check for explicit contradictions
            for pattern in negation_patterns:
                if re.search(pattern, fact_lower) and self._compute_semantic_similarity(claim, fact.fact_text) >= threshold:
                    contradicting_indices.append(i)
                    break
        
        return contradicting_indices
    
    def _compute_support_score(self, claims: List[GroundingClaim]) -> float:
        """
        Compute overall support score for all claims
        
        Args:
            claims: List of claims with their support information
            
        Returns:
            Overall support score between 0 and 1
        """
        if not claims:
            return 0.0
        
        # Count claims that require grounding check
        grounding_required_claims = [c for c in claims if c.grounding_check_required]
        
        if not grounding_required_claims:
            return 1.0  # No claims require grounding, so fully supported
        
        # Count supported claims (those with citations)
        supported_claims = [c for c in grounding_required_claims if c.citations]
        
        return len(supported_claims) / len(grounding_required_claims)
    
    def _compute_contradiction_score(self, claims: List[GroundingClaim]) -> float:
        """
        Compute overall contradiction score for all claims
        
        Args:
            claims: List of claims with their contradiction information
            
        Returns:
            Overall contradiction score between 0 and 1
        """
        if not claims:
            return 0.0
        
        # Count claims that require grounding check
        grounding_required_claims = [c for c in claims if c.grounding_check_required]
        
        if not grounding_required_claims:
            return 0.0  # No claims require grounding, so no contradictions
        
        # Count contradicted claims (those with anti-citations)
        contradicted_claims = [c for c in grounding_required_claims if c.anti_citations]
        
        return len(contradicted_claims) / len(grounding_required_claims)
    
    def _compute_helpfulness_score(self, answer_candidate: str, prompt: str = None) -> float:
        """
        Compute helpfulness score for the answer candidate
        
        Args:
            answer_candidate: The answer to evaluate
            prompt: The original prompt/question (optional)
            
        Returns:
            Helpfulness score between 0 and 1
        """
        if not prompt:
            # Without a prompt, we can only do basic helpfulness checks
            score = 0.5
            
            # Check for completeness indicators
            if len(answer_candidate.split()) > 10:
                score += 0.2
            
            # Check for structure (multiple sentences)
            sentences = re.split(r'[.!?]+', answer_candidate)
            if len(sentences) > 2:
                score += 0.1
            
            # Check for specific information (numbers, names, dates)
            if re.search(r'\b\d{4}\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b', answer_candidate):
                score += 0.2
            
            return min(1.0, score)
        
        # With a prompt, we can do semantic similarity
        similarity = self._compute_semantic_similarity(answer_candidate, prompt)
        
        # Adjust based on answer length and structure
        word_count = len(answer_candidate.split())
        if word_count < 5:
            similarity *= 0.5  # Too short
        elif word_count > 200:
            similarity *= 0.8  # Potentially too verbose
        
        return similarity
    
    def _create_cited_chunks(self, facts: List[GroundingFact], 
                           cited_indices: set) -> List[CitedChunk]:
        """
        Create cited chunks from facts that were referenced
        
        Args:
            facts: List of all facts
            cited_indices: Set of fact indices that were cited
            
        Returns:
            List of CitedChunk objects
        """
        cited_chunks = []
        
        for index in cited_indices:
            if index < len(facts):
                fact = facts[index]
                chunk = CitedChunk(
                    fact_index=index,
                    chunk_text=fact.fact_text,
                    attributes=fact.attributes.copy()
                )
                cited_chunks.append(chunk)
        
        return cited_chunks
    
    async def check_grounding(self, answer_candidate: str, facts: List[GroundingFact],
                            grounding_spec: GroundingSpec = None, 
                            prompt: str = None) -> GroundingResponse:
        """
        Check grounding of an answer candidate against a set of facts
        
        Args:
            answer_candidate: The text to check for grounding
            facts: List of facts to check against
            grounding_spec: Configuration for the grounding check
            prompt: Original prompt/question for helpfulness scoring
            
        Returns:
            GroundingResponse with scores and citations
        """
        if grounding_spec is None:
            grounding_spec = GroundingSpec()
        
        # Extract claims from answer candidate
        claims = self._extract_claims(answer_candidate)
        
        # Process each claim
        all_cited_indices = set()
        
        for claim in claims:
            if not claim.grounding_check_required:
                continue
            
            # Find supporting facts
            supporting_indices = self._find_supporting_facts(
                claim.text, facts, grounding_spec.citation_threshold
            )
            claim.citations = supporting_indices
            all_cited_indices.update(supporting_indices)
            
            # Find contradicting facts if enabled
            if grounding_spec.enable_anti_citations:
                contradicting_indices = self._find_contradicting_facts(
                    claim.text, facts, grounding_spec.anti_citation_threshold
                )
                claim.anti_citations = contradicting_indices
            
            # Compute claim-level support score if enabled
            if grounding_spec.enable_claim_level_score:
                if supporting_indices:
                    # Average similarity with supporting facts
                    similarities = []
                    for idx in supporting_indices:
                        sim = self._compute_semantic_similarity(claim.text, facts[idx].fact_text)
                        similarities.append(sim)
                    claim.support_score = sum(similarities) / len(similarities)
                else:
                    claim.support_score = 0.0
        
        # Compute overall scores
        support_score = self._compute_support_score(claims)
        contradiction_score = 0.0
        helpfulness_score = 0.0
        
        if grounding_spec.enable_anti_citations:
            contradiction_score = self._compute_contradiction_score(claims)
        
        if grounding_spec.enable_helpfulness_score:
            helpfulness_score = self._compute_helpfulness_score(answer_candidate, prompt)
        
        # Create cited chunks
        cited_chunks = self._create_cited_chunks(facts, all_cited_indices)
        
        return GroundingResponse(
            support_score=support_score,
            contradiction_score=contradiction_score,
            helpfulness_score=helpfulness_score,
            claims=claims,
            cited_chunks=cited_chunks
        )
    
    def format_grounding_response(self, response: GroundingResponse, 
                                include_details: bool = True) -> str:
        """
        Format grounding response for display
        
        Args:
            response: The grounding response to format
            include_details: Whether to include detailed claim information
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append(f"**Grounding Check Results**")
        lines.append(f"Support Score: {response.support_score:.2f}")
        
        if response.contradiction_score > 0:
            lines.append(f"Contradiction Score: {response.contradiction_score:.2f}")
        
        if response.helpfulness_score > 0:
            lines.append(f"Helpfulness Score: {response.helpfulness_score:.2f}")
        
        if include_details and response.claims:
            lines.append("\n**Claims Analysis:**")
            for i, claim in enumerate(response.claims):
                if not claim.grounding_check_required:
                    lines.append(f"{i}. \"{claim.text}\" - No grounding check required")
                else:
                    citation_info = f"Citations: {claim.citations}" if claim.citations else "Citations: None"
                    if claim.support_score > 0:
                        citation_info += f" (Score: {claim.support_score:.2f})"
                    lines.append(f"{i}. \"{claim.text}\" - {citation_info}")
                    
                    if claim.anti_citations:
                        lines.append(f"   Anti-citations: {claim.anti_citations}")
        
        if response.cited_chunks:
            lines.append(f"\n**Cited Sources ({len(response.cited_chunks)} total):**")
            for i, chunk in enumerate(response.cited_chunks):
                chunk_preview = chunk.chunk_text[:100] + "..." if len(chunk.chunk_text) > 100 else chunk.chunk_text
                lines.append(f"[{chunk.fact_index}] {chunk_preview}")
                if chunk.attributes:
                    attr_str = ", ".join([f"{k}: {v}" for k, v in chunk.attributes.items()])
                    lines.append(f"    Attributes: {attr_str}")
        
        return "\n".join(lines)
    
    def to_dict(self, response: GroundingResponse) -> Dict[str, Any]:
        """
        Convert grounding response to dictionary format
        
        Args:
            response: The grounding response to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "support_score": response.support_score,
            "contradiction_score": response.contradiction_score,
            "helpfulness_score": response.helpfulness_score,
            "claims": [
                {
                    "text": claim.text,
                    "start_pos": claim.start_pos,
                    "end_pos": claim.end_pos,
                    "support_score": claim.support_score,
                    "citations": claim.citations,
                    "anti_citations": claim.anti_citations,
                    "grounding_check_required": claim.grounding_check_required
                }
                for claim in response.claims
            ],
            "cited_chunks": [
                {
                    "fact_index": chunk.fact_index,
                    "chunk_text": chunk.chunk_text,
                    "attributes": chunk.attributes
                }
                for chunk in response.cited_chunks
            ]
        }


# Example usage and testing functions
async def example_usage():
    """Example of how to use the GroundingManager"""
    
    # Initialize the manager
    grounding_manager = GroundingManager()
    
    # Create some example facts
    facts = [
        GroundingFact(
            fact_text="Titanic is a 1997 American epic romantic disaster movie. It was directed, written, and co-produced by James Cameron. The movie is about the 1912 sinking of the RMS Titanic. It stars Kate Winslet and Leonardo DiCaprio. The movie was released on December 19, 1997. It received positive critical reviews. The movie won 11 Academy Awards, and was nominated for fourteen total Academy Awards.",
            attributes={"author": "Simple Wikipedia"}
        ),
        GroundingFact(
            fact_text="James Cameron's \"Titanic\" is an epic, action-packed romance set against the ill-fated maiden voyage of the R.M.S. Titanic; the pride and joy of the White Star Line and, at the time, the largest moving object ever built. She was the most luxurious liner of her era -- the \"ship of dreams\" -- which ultimately carried over 1,500 people to their death in the ice cold waters of the North Atlantic in the early hours of April 15, 1912.",
            attributes={"author": "Rotten Tomatoes"}
        )
    ]
    
    # Test answer candidate
    answer_candidate = "Titanic was directed by James Cameron. It was released in 1997."
    
    # Create grounding spec with all features enabled
    grounding_spec = GroundingSpec(
        citation_threshold=0.6,
        enable_claim_level_score=True,
        enable_anti_citations=True,
        enable_helpfulness_score=True
    )
    
    # Check grounding
    response = await grounding_manager.check_grounding(
        answer_candidate=answer_candidate,
        facts=facts,
        grounding_spec=grounding_spec,
        prompt="Who directed Titanic and when was it released?"
    )
    
    # Format and print results
    formatted_response = grounding_manager.format_grounding_response(response)
    print(formatted_response)
    
    return response


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

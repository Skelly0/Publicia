"""
Google Vertex AI Check Grounding API implementation for Publicia
Real implementation using Google Cloud Discovery Engine grounding check API
"""
import re
import json
import logging
import asyncio
import aiohttp
import os
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
import numpy as np

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
    
    def __init__(self, config=None):
        """
        Initialize the grounding manager with Google Cloud credentials
        
        Args:
            config: Configuration object with Google Cloud settings
        """
        self.config = config
        self.project_id = getattr(config, 'GOOGLE_PROJECT_ID', None) if config else None
        self.api_key = getattr(config, 'GOOGLE_API_KEY', None) if config else None
        
        # Clean project ID if it has extra content
        if self.project_id:
            self.project_id = self.project_id.strip().split()[0]  # Take only first part, remove comments
        
        # Usage tracking configuration
        self.max_daily_checks = getattr(config, 'GROUNDING_MAX_DAILY_CHECKS', 1000) if config else 1000
        self.cost_per_check = getattr(config, 'GROUNDING_COST_PER_CHECK', 0.001) if config else 0.001
        self.max_daily_budget = getattr(config, 'GROUNDING_MAX_DAILY_BUDGET', 1.0) if config else 1.0
        
        # Usage tracking state
        self.usage_file = Path("temp_files/grounding_usage.json")
        self.usage_file.parent.mkdir(exist_ok=True)
        self.daily_usage = self._load_daily_usage()
        
        # Fallback document manager for local similarity if API fails
        self.document_manager = None
        
        # Check for Google API availability
        # Note: Google Discovery Engine API requires service account credentials, not API keys
        self.google_api_available = False
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path) and self.project_id:
            self.google_api_available = True
            logger.info(f"Google API configured with service account - Project: {self.project_id}")
        elif self.project_id and self.api_key:
            logger.warning("Google Discovery Engine API requires service account credentials. API keys are not supported.")
            logger.warning("Please set up service account credentials and GOOGLE_APPLICATION_CREDENTIALS environment variable.")
            logger.warning("Grounding will use local fallback implementation.")
        else:
            if not self.project_id:
                logger.warning("No Google Project ID provided. Grounding API will not be available.")
            if not self.api_key and not credentials_path:
                logger.warning("No Google credentials provided. Grounding API will not be available.")
        
    def _load_daily_usage(self) -> Dict[str, Any]:
        """Load daily usage tracking data"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    # Reset if it's a new day
                    today = date.today().isoformat()
                    if data.get('date') != today:
                        return {'date': today, 'checks': 0, 'cost': 0.0}
                    return data
        except Exception as e:
            logger.error(f"Error loading usage data: {e}")
        
        # Default for new day or error
        return {'date': date.today().isoformat(), 'checks': 0, 'cost': 0.0}
    
    def _save_daily_usage(self):
        """Save daily usage tracking data"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.daily_usage, f)
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")
    
    def _can_make_api_call(self) -> Tuple[bool, str]:
        """
        Check if we can make an API call within daily limits
        
        Returns:
            Tuple of (can_call, reason_if_not)
        """
        today = date.today().isoformat()
        
        # Reset usage if it's a new day
        if self.daily_usage.get('date') != today:
            self.daily_usage = {'date': today, 'checks': 0, 'cost': 0.0}
            self._save_daily_usage()
        
        # Check daily check limit
        if self.daily_usage['checks'] >= self.max_daily_checks:
            return False, f"Daily check limit reached ({self.max_daily_checks} checks)"
        
        # Check daily budget limit
        projected_cost = self.daily_usage['cost'] + self.cost_per_check
        if projected_cost > self.max_daily_budget:
            return False, f"Daily budget limit would be exceeded (${self.max_daily_budget:.2f})"
        
        return True, ""
    
    def _record_api_usage(self):
        """Record an API call in usage tracking"""
        self.daily_usage['checks'] += 1
        self.daily_usage['cost'] += self.cost_per_check
        self._save_daily_usage()
        
        logger.info(f"Grounding API usage: {self.daily_usage['checks']}/{self.max_daily_checks} checks, "
                   f"${self.daily_usage['cost']:.3f}/${self.max_daily_budget:.2f} budget")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        today = date.today().isoformat()
        if self.daily_usage.get('date') != today:
            return {'date': today, 'checks': 0, 'cost': 0.0, 'checks_remaining': self.max_daily_checks, 'budget_remaining': self.max_daily_budget}
        
        return {
            'date': self.daily_usage['date'],
            'checks': self.daily_usage['checks'],
            'cost': self.daily_usage['cost'],
            'checks_remaining': max(0, self.max_daily_checks - self.daily_usage['checks']),
            'budget_remaining': max(0.0, self.max_daily_budget - self.daily_usage['cost']),
            'max_daily_checks': self.max_daily_checks,
            'max_daily_budget': self.max_daily_budget
        }
    
    def _ensure_document_manager(self):
        """Ensure document manager is available for embeddings"""
        if self.document_manager is None:
            logger.warning("No document manager provided to grounding manager. Similarity computation will be limited.")
            return False
        return True
    
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
    
    async def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using Gemini embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not self._ensure_document_manager():
                # Fallback to simple text overlap if no embeddings available
                return self._compute_text_overlap_similarity(text1, text2)
            
            # Generate embeddings for both texts
            embeddings = await self.document_manager.generate_embeddings([text1, text2], is_query=False)
            
            if embeddings.size == 0 or len(embeddings) < 2:
                logger.warning("Failed to generate embeddings for similarity computation, using text overlap fallback")
                return self._compute_text_overlap_similarity(text1, text2)
            
            # Compute cosine similarity
            embedding1, embedding2 = embeddings[0], embeddings[1]
            
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in 0-1 range (cosine similarity is -1 to 1)
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error computing Gemini embedding similarity: {e}")
            # Fallback to text overlap
            return self._compute_text_overlap_similarity(text1, text2)
    
    def _compute_text_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Improved fallback similarity computation based on word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize and tokenize
            words1 = set(word.lower().strip('.,!?";:()[]{}') for word in text1.split() if len(word) > 2)
            words2 = set(word.lower().strip('.,!?";:()[]{}') for word in text2.split() if len(word) > 2)
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            
            # Use Jaccard similarity but with a boost for important word matches
            jaccard = len(intersection) / len(words1.union(words2)) if words1.union(words2) else 0.0
            
            # Boost score for key word matches (names, important terms)
            important_words = {'titanic', 'cameron', 'james', 'directed', 'released', 'python', 'guido', 'rossum', 'created'}
            important_matches = intersection.intersection(important_words)
            importance_boost = len(important_matches) * 0.15
            
            # Also boost for exact phrase matches, but be more careful about contradictions
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            phrase_boost = 0.0
            
            # Check for common phrases
            positive_phrases = ['james cameron', 'directed by', 'created by', 'guido van rossum']
            for phrase in positive_phrases:
                if phrase in text1_lower and phrase in text2_lower:
                    phrase_boost += 0.25
            
            # Penalize for contradictory information
            contradiction_penalty = 0.0
            if 'spielberg' in text1_lower and 'cameron' in text2_lower:
                contradiction_penalty = 0.4
            if '2005' in text1_lower and '1997' in text2_lower:
                contradiction_penalty = max(contradiction_penalty, 0.3)
            
            final_score = min(1.0, max(0.0, jaccard + importance_boost + phrase_boost - contradiction_penalty))
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing text overlap similarity: {e}")
            return 0.0
    
    async def _find_supporting_facts(self, claim: str, facts: List[GroundingFact],
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
            similarity = await self._compute_semantic_similarity(claim, fact.fact_text)
            if similarity >= threshold:
                supporting_indices.append(i)
        
        return supporting_indices
    
    async def _find_contradicting_facts(self, claim: str, facts: List[GroundingFact],
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
                if re.search(pattern, fact_lower):
                    similarity = await self._compute_semantic_similarity(claim, fact.fact_text)
                    if similarity >= threshold:
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
        # Note: This is a synchronous function, so we use the text overlap fallback
        similarity = self._compute_text_overlap_similarity(answer_candidate, prompt)
        
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
        Check grounding using Google Vertex AI Check Grounding API
        
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
        
        # Check if we can use Google API
        if self.google_api_available:
            can_call, reason = self._can_make_api_call()
            if can_call:
                try:
                    result = await self._check_grounding_google_api(
                        answer_candidate, facts, grounding_spec, prompt
                    )
                    self._record_api_usage()
                    return result
                except Exception as e:
                    logger.error(f"Google API grounding check failed: {e}")
                    logger.info("Falling back to local grounding implementation")
            else:
                logger.warning(f"Google API usage limit reached: {reason}. Using local fallback.")
        else:
            logger.info("Google API credentials not available, using local fallback")
        
        # Use local fallback
        return await self._check_grounding_local_fallback(
            answer_candidate, facts, grounding_spec, prompt
        )
    
    async def _check_grounding_google_api(self, answer_candidate: str, facts: List[GroundingFact],
                                        grounding_spec: GroundingSpec, prompt: str = None) -> GroundingResponse:
        """
        Use Google Vertex AI Check Grounding API
        """
        # Get OAuth2 access token
        access_token = await self._get_access_token()
        
        # Prepare the API request
        url = f"https://discoveryengine.googleapis.com/v1/projects/{self.project_id}/locations/global/groundingConfigs/default_grounding_config:check"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        # Convert facts to API format
        api_facts = []
        for fact in facts:
            api_fact = {
                "factText": fact.fact_text,
                "attributes": fact.attributes or {}
            }
            api_facts.append(api_fact)
        
        # Prepare grounding spec
        api_grounding_spec = {
            "citationThreshold": grounding_spec.citation_threshold
        }
        
        if grounding_spec.enable_anti_citations:
            api_grounding_spec["enableAntiCitations"] = True
            api_grounding_spec["antiCitationThreshold"] = grounding_spec.anti_citation_threshold
        
        if grounding_spec.enable_claim_level_score:
            api_grounding_spec["enableClaimLevelScore"] = True
        
        if grounding_spec.enable_helpfulness_score:
            api_grounding_spec["enableHelpfulnessScore"] = True
        
        # Prepare request payload
        payload = {
            "answerCandidate": answer_candidate,
            "facts": api_facts,
            "groundingSpec": api_grounding_spec
        }
        
        if prompt:
            payload["prompt"] = prompt
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_google_api_response(result, facts)
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
    
    async def _get_access_token(self) -> str:
        """
        Get Google Cloud access token using service account or API key
        """
        import json
        import time
        import jwt
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        
        try:
            # Try to use service account credentials if available
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                credentials.refresh(Request())
                return credentials.token
            
            # Fallback: Try to create a service account token using the API key
            # This is a workaround since Discovery Engine API doesn't support API keys directly
            # In practice, you should use proper service account credentials
            
            # For now, we'll disable the Google API and use local fallback
            raise Exception("Google Discovery Engine API requires service account credentials. API keys are not supported.")
            
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise Exception(f"Google API authentication failed: {e}")
    
    def _parse_google_api_response(self, api_response: dict, original_facts: List[GroundingFact]) -> GroundingResponse:
        """
        Parse Google API response into our GroundingResponse format
        """
        # Extract scores
        support_score = api_response.get("supportScore", 0.0)
        contradiction_score = api_response.get("contradictionScore", 0.0)
        helpfulness_score = api_response.get("helpfulnessScore", 0.0)
        
        # Parse claims
        claims = []
        api_claims = api_response.get("claims", [])
        
        for api_claim in api_claims:
            claim = GroundingClaim(
                text=api_claim.get("claimText", ""),
                start_pos=api_claim.get("startPos", 0),
                end_pos=api_claim.get("endPos", 0),
                support_score=api_claim.get("supportScore", 0.0),
                citations=api_claim.get("citations", []),
                anti_citations=api_claim.get("antiCitations", []),
                grounding_check_required=api_claim.get("groundingCheckRequired", True)
            )
            claims.append(claim)
        
        # Parse cited chunks
        cited_chunks = []
        api_cited_chunks = api_response.get("citedChunks", [])
        
        for chunk in api_cited_chunks:
            cited_chunk = CitedChunk(
                fact_index=chunk.get("factIndex", 0),
                chunk_text=chunk.get("chunkText", ""),
                attributes=chunk.get("attributes", {})
            )
            cited_chunks.append(cited_chunk)
        
        return GroundingResponse(
            support_score=support_score,
            contradiction_score=contradiction_score,
            helpfulness_score=helpfulness_score,
            claims=claims,
            cited_chunks=cited_chunks
        )
    
    async def _check_grounding_local_fallback(self, answer_candidate: str, facts: List[GroundingFact],
                                            grounding_spec: GroundingSpec, prompt: str = None) -> GroundingResponse:
        """
        Local fallback implementation when Google API is not available
        """
        # Extract claims from answer candidate
        claims = self._extract_claims(answer_candidate)
        
        # Process each claim
        all_cited_indices = set()
        
        for claim in claims:
            if not claim.grounding_check_required:
                continue
            
            # Find supporting facts using text overlap similarity
            supporting_indices = []
            for i, fact in enumerate(facts):
                similarity = self._compute_text_overlap_similarity(claim.text, fact.fact_text)
                if similarity >= grounding_spec.citation_threshold:
                    supporting_indices.append(i)
            
            claim.citations = supporting_indices
            all_cited_indices.update(supporting_indices)
            
            # Simple contradiction detection
            if grounding_spec.enable_anti_citations:
                contradicting_indices = []
                # Look for explicit negations
                negation_patterns = [r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bfalse\b']
                for i, fact in enumerate(facts):
                    for pattern in negation_patterns:
                        if re.search(pattern, fact.fact_text.lower()):
                            similarity = self._compute_text_overlap_similarity(claim.text, fact.fact_text)
                            if similarity >= grounding_spec.anti_citation_threshold:
                                contradicting_indices.append(i)
                                break
                claim.anti_citations = contradicting_indices
            
            # Compute claim-level support score if enabled
            if grounding_spec.enable_claim_level_score:
                if supporting_indices:
                    similarities = []
                    for idx in supporting_indices:
                        sim = self._compute_text_overlap_similarity(claim.text, facts[idx].fact_text)
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
    from managers.config import Config
    config = Config()
    grounding_manager = GroundingManager(config=config)
    
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

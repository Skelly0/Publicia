"""
Configuration management for Publicia
"""
import os
import logging
from dotenv import load_dotenv
from .doc_tracking_channels import DocTrackingChannelManager

logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # Configure models with defaults
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'google/gemini-2.5-flash')
        self.DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'qwen/qwq-32b')
        self.CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'google/gemini-2.5-flash')
        self.AGENTIC_MODEL = os.getenv('AGENTIC_MODEL', 'openai/o4-mini')
        
        # New embedding configuration
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-004')
        self.EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '0'))
        
        # Chunk size configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '300'))  # Default to 300 words per chunk
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '30'))  # Default to 125 words overlap
        
        # TOP_K configuration with multiplier
        self.TOP_K = int(os.getenv('TOP_K', '5'))
        self.MAX_TOP_K = int(os.getenv('MAX_TOP_K', '50'))
        self.VIEW_CHUNK_LIMIT = int(os.getenv('VIEW_CHUNK_LIMIT', '5'))
        self.MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', '50'))

        self.TOP_K_MULTIPLIER = float(os.getenv('TOP_K_MULTIPLIER', '1'))  # Default to no change

        self.MODEL_TOP_K = {
            # DeepSeek models
            "deepseek/deepseek-chat-v3-0324": 15, # User specified value
            "deepseek/deepseek-chat-v3-0324:floor": 15, # User specified value
            "deepseek/deepseek-chat-v3-0324": 15, # User specified value
            "deepseek/deepseek-chat": 15,
            "deepseek/deepseek-r1": 15,
            "deepseek/deepseek-r1-distill-llama-70b": 14,
            "deepseek/deepseek-r1:floor": 10,
            "deepseek/deepseek-r1:nitro": 7,
            "deepseek/deepseek-r1-0528": 15,
            "deepseek/deepseek-r1-0528:floor": 1150,
            # Gemini models 
            "google/gemini-2.5-flash": 18,
            "google/gemini-2.5-flash:thinking": 16,
            "google/gemini-2.0-pro-exp-02-05": 20,
            "google/gemini-2.5-pro": 10, # Added new model
            # Nous Hermes models
            "nousresearch/hermes-3-llama-3.1-405b": 9,
            # Claude models
            "anthropic/claude-3.5-haiku": 13,
            "anthropic/claude-3.5-haiku": 13,
            "anthropic/claude-sonnet-4": 5,
            "anthropic/claude-sonnet-4": 5,
            "anthropic/claude-3.7-sonnet": 5,
            "anthropic/claude-3.7-sonnet": 5,
            # Qwen models
            "qwen/qwq-32b": 20,
            "qwen/qwen3-235b-a22b-thinking-2507": 17,
            # Testing models
            "thedrummer/unslopnemo-12b": 12,
            "eva-unit-01/eva-qwen-2.5-72b": 9,
            # New models
            "latitudegames/wayfarer-large-70b-llama-3.3": 9,  # good balance for storytelling
            "thedrummer/anubis-pro-105b-v1": 8,  # slightly smaller for this massive model
            # Gemini embedding model
            "models/text-embedding-004": 20,  # Optimized for larger chunks
            # Microsoft Models
            "microsoft/phi-4-multimodal-instruct": 15,
            # Meta-Llama Maverick and fallbacks
            "meta-llama/llama-4-maverick:floor": 9,
            "meta-llama/llama-4-maverick": 9,
            "meta-llama/llama-4-scout": 9,
            # OpenAI Models
            "openai/gpt-4.1-mini": 11,
            "openai/gpt-4.1-nano": 17,
            "openai/gpt-5-mini": 8,
            "openai/o4-mini": 8,
            "openai/gpt-oss-120b": 18,
            # X AI Models
            "x-ai/grok-3-mini-beta": 20,
            # MiniMax Models
            "minimax/minimax-m1": 20,
            # Moonshot AI Models
            "moonshotai/kimi-k2": 20,
            # Zhipu AI Models
            "z-ai/glm-4.5": 20,
            "switchpoint/router": 12,
        }
        
        # Validate required environment variables
        self._validate_config()
        
        # Add timeout settings
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', '180'))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', '10'))

        self.MODEL_PROVIDERS = {
            "deepseek/deepseek-r1": {
                "order": ["Minimax", "Nebius", "DeepInfra"]
            },
            "eva-unit-01/eva-qwen-2.5-72b": {
                "order": ["Parasail"]
            },
            "qwen/qwq-32b:floor": {
                "order": ["Groq", "DeepInfra", "Hyperbolic"]
            },
            "minimax/minimax-m1": {
                "order": ["minimax", "novita"]
            },
            # Add any other model variants that need custom provider ordering
        }

        self.TEMPERATURE_BASE = float(os.getenv('TEMPERATURE_BASE', '0.1'))
        self.TEMPERATURE_MIN = float(os.getenv('TEMPERATURE_MIN', '0.0'))
        self.TEMPERATURE_MAX = float(os.getenv('TEMPERATURE_MAX', '0.4'))

        self.RERANKING_ENABLED = bool(os.getenv('RERANKING_ENABLED', 'False').lower() in ('true', '1', 'yes'))
        self.RERANKING_CANDIDATES = int(os.getenv('RERANKING_CANDIDATES', '20'))  # Number of initial candidates
        self.RERANKING_MIN_SCORE = float(os.getenv('RERANKING_MIN_SCORE', '0.5'))  # Minimum score threshold

        self.RERANKING_FILTER_MODE = os.getenv('RERANKING_FILTER_MODE', 'strict')  # 'strict', 'dynamic', or 'topk'

        # BM25 search weighting configuration
        self.BM25_WEIGHT = float(os.getenv('BM25_WEIGHT', '0.25'))  # Default to 0.25 (25% BM25, 75% embedding) - improved for factual queries

        # Keyword checking configuration
        self.KEYWORD_CHECK_CHUNK_LIMIT = int(os.getenv('KEYWORD_CHECK_CHUNK_LIMIT', '5')) # Number of chunks to check for keywords

        # Permission settings (comma-separated IDs)
        self.ALLOWED_USER_IDS = [int(uid.strip()) for uid in os.getenv('ALLOWED_USER_IDS', '').split(',') if uid.strip().isdigit()]
        self.ALLOWED_ROLE_IDS = [int(rid.strip()) for rid in os.getenv('ALLOWED_ROLE_IDS', '').split(',') if rid.strip().isdigit()]

        # Google Doc tracking channels handled via JSON file
        self.doc_channel_manager = DocTrackingChannelManager()
        self.DOC_TRACKING_CHANNEL_IDS = self.doc_channel_manager.get_channels()

        # Migrate legacy .env setting if present
        doc_tracking_channel_id_str = os.getenv('DOC_TRACKING_CHANNEL_ID')
        if doc_tracking_channel_id_str and doc_tracking_channel_id_str.isdigit():
            channel_id = int(doc_tracking_channel_id_str)
            if self.doc_channel_manager.add_channel(channel_id):
                logger.info(
                    "Migrated DOC_TRACKING_CHANNEL_ID from .env to doc_tracking_channels.json"
                )
            if channel_id not in self.DOC_TRACKING_CHANNEL_IDS:
                self.DOC_TRACKING_CHANNEL_IDS.append(channel_id)
        elif doc_tracking_channel_id_str:
            logger.warning(
                f"Invalid DOC_TRACKING_CHANNEL_ID: '{doc_tracking_channel_id_str}'. Must be an integer."
            )

        # Auto-processing setting for Google Docs
        self.AUTO_PROCESS_GOOGLE_DOCS = bool(os.getenv('AUTO_PROCESS_GOOGLE_DOCS', 'False').lower() in ('true', '1', 'yes'))

        # Keyword database system enable/disable setting
        self.KEYWORD_DATABASE_ENABLED = bool(os.getenv('KEYWORD_DATABASE_ENABLED', 'True').lower() in ('true', '1', 'yes'))

        # Toggle LLM access to the internal document list
        self.DOCUMENT_LIST_ENABLED = bool(os.getenv('DOCUMENT_LIST_ENABLED', 'True').lower() in ('true', '1', 'yes'))

        # Contextualization settings
        self.CONTEXTUALIZATION_ENABLED = bool(os.getenv('CONTEXTUALIZATION_ENABLED', 'True').lower() in ('true', '1', 'yes'))
        self.MAX_WORDS_FOR_CONTEXT = int(os.getenv('MAX_WORDS_FOR_CONTEXT', '20000'))
        self.USE_CONTEXTUALISED_CHUNKS = bool(os.getenv('USE_CONTEXTUALISED_CHUNKS', 'True').lower() in ('true', '1', 'yes'))
        self.CHANNEL_CONTEXTUALIZATION_ENABLED = bool(os.getenv('CHANNEL_CONTEXTUALIZATION_ENABLED', 'True').lower() in ('true', '1', 'yes'))

    def get_reranking_settings_for_query(self, query: str):
        """Get adaptive reranking settings based on query complexity."""
        complex_indicators = [
            'analysis', 'analyze', 'detailed', 'comprehensive', 'intersection',
            'relationship', 'compare', 'contrast', 'philosophy', 'theology',
            'write about', 'explain in detail', 'discuss', 'elaborate',
            'write a', 'provide a', 'give me a', 'tell me',
            'describe', 'overview', 'summary', 'breakdown', 'examination',
            'exploration', 'investigation', 'study', 'research', 'deep dive'
        ]
        
        is_complex = any(indicator in query.lower() for indicator in complex_indicators)
        
        if is_complex:
            logger.info("Detected complex query, using lenient reranking settings")
            return {
                'min_score': 0.15,      # Even lower threshold for better recall
                'filter_mode': 'topk',  # No score filtering (was 'strict')
                'candidates': 40        # Even more candidates for better selection
            }
        else:
            return {
                'min_score': self.RERANKING_MIN_SCORE,
                'filter_mode': self.RERANKING_FILTER_MODE,
                'candidates': self.RERANKING_CANDIDATES
            }
        
   
    def get_provider_config(self, model: str):
        """Get provider config for a specific model."""
        # Extract base model without suffixes like  or :nitro
        base_model = model.split(':')[0] if ':' in model else model
        
        # First try exact match, then try base model
        return self.MODEL_PROVIDERS.get(model) or self.MODEL_PROVIDERS.get(base_model)
        
    def get_top_k_for_model(self, model: str) -> int:
        """Get the top_k value for a specific model, applying the configured multiplier."""
        # First try to match the exact model name (including any suffixes)
        if model in self.MODEL_TOP_K:
            base_top_k = self.MODEL_TOP_K[model]
        else:
            # If not found, extract the base model name and try that
            base_model = model.split(':')[0] if ':' in model else model
            base_top_k = self.MODEL_TOP_K.get(base_model, self.TOP_K)
        
        # Apply the multiplier and round to the nearest integer
        # Ensure we always return at least 1 result
        adjusted_top_k = max(1, round(base_top_k * self.TOP_K_MULTIPLIER))
        
        # Log the adjustment if multiplier isn't 1.0
        if self.TOP_K_MULTIPLIER != 1.0:
            logger.info(f"Adjusted top_k for {model} from {base_top_k} to {adjusted_top_k} (multiplier: {self.TOP_K_MULTIPLIER})")
    
        return adjusted_top_k
    
    def _validate_config(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            'DISCORD_BOT_TOKEN',
            'OPENROUTER_API_KEY',
            'GOOGLE_API_KEY' # Added GOOGLE_API_KEY as it's needed for embeddings
            # LLM_MODEL and CLASSIFIER_MODEL are not required as they have defaults
        ]
        
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

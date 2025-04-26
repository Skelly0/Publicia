"""
Configuration management for Publicia
"""
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the bot."""
    
    def __init__(self):
        load_dotenv()
        self.DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # Configure models with defaults
        self.LLM_MODEL = os.getenv('LLM_MODEL', 'google/gemini-2.5-flash-preview')
        self.DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'qwen/qwq-32b')
        self.CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'google/gemini-2.5-flash-preview')
        
        # New embedding configuration
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-004')
        self.EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '0'))
        
        # Chunk size configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '300'))  # Default to 750 words per chunk
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '30'))  # Default to 125 words overlap
        
        # TOP_K configuration with multiplier
        self.TOP_K = int(os.getenv('TOP_K', '5'))
        self.MAX_TOP_K = int(os.getenv('MAX_TOP_K', '20'))
        self.TOP_K_MULTIPLIER = float(os.getenv('TOP_K_MULTIPLIER', '1'))  # Default to no change

        self.MODEL_TOP_K = {
            # DeepSeek models
            "deepseek/deepseek-chat-v3-0324:free": 10, # User specified value
            "deepseek/deepseek-chat-v3-0324:floor": 10, # User specified value
            "deepseek/deepseek-chat-v3-0324": 10, # User specified value
            "deepseek/deepseek-chat": 10,
            "deepseek/deepseek-r1:free": 20,
            "deepseek/deepseek-r1": 10,
            "deepseek/deepseek-r1-distill-llama-70b": 14,
            "deepseek/deepseek-r1:floor": 10,
            "deepseek/deepseek-r1:nitro": 7,
            # Gemini models 
            "google/gemini-2.5-flash-preview": 18,
            "google/gemini-2.5-flash-preview:thinking": 16,
            "google/gemini-2.0-pro-exp-02-05:free": 20,
            "google/gemini-2.5-pro-exp-03-25:free": 20, # Added new model
            # Nous Hermes models
            "nousresearch/hermes-3-llama-3.1-405b": 9,
            # Claude models
            "anthropic/claude-3.5-haiku:beta": 9,
            "anthropic/claude-3.5-haiku": 9,
            "anthropic/claude-3.5-sonnet:beta": 5,
            "anthropic/claude-3.5-sonnet": 5,
            "anthropic/claude-3.7-sonnet:beta": 5,
            "anthropic/claude-3.7-sonnet": 5,
            # Qwen models
            "qwen/qwq-32b:free": 20,
            "qwen/qwq-32b": 16, 
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
            # X AI Models
            "x-ai/grok-3-mini-beta": 14,
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
                "order": ["Groq", "Nineteen", "DeepInfra", "Nebius", "Hyperbolic"]
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

        # Permission settings (comma-separated IDs)
        self.ALLOWED_USER_IDS = [int(uid.strip()) for uid in os.getenv('ALLOWED_USER_IDS', '').split(',') if uid.strip().isdigit()]
        self.ALLOWED_ROLE_IDS = [int(rid.strip()) for rid in os.getenv('ALLOWED_ROLE_IDS', '').split(',') if rid.strip().isdigit()]

        # Document tracking channel ID
        doc_tracking_channel_id_str = os.getenv('DOC_TRACKING_CHANNEL_ID')
        self.DOC_TRACKING_CHANNEL_ID = None
        if doc_tracking_channel_id_str and doc_tracking_channel_id_str.isdigit():
            self.DOC_TRACKING_CHANNEL_ID = int(doc_tracking_channel_id_str)
        elif doc_tracking_channel_id_str:
            logger.warning(f"Invalid DOC_TRACKING_CHANNEL_ID: '{doc_tracking_channel_id_str}'. Must be an integer. Disabling feature.")
        
        # Validate temperature settings
        if not (0 <= self.TEMPERATURE_MIN <= self.TEMPERATURE_BASE <= self.TEMPERATURE_MAX <= 1):
            logger.warning(f"Invalid temperature settings: MIN({self.TEMPERATURE_MIN}), BASE({self.TEMPERATURE_BASE}), MAX({self.TEMPERATURE_MAX})")
            logger.warning("Temperatures should satisfy: 0 ≤ MIN ≤ BASE ≤ MAX ≤ 1")
            # Fall back to defaults if settings are invalid
            self.TEMPERATURE_BASE = 0.1 
            self.TEMPERATURE_MIN = 0.0
            self.TEMPERATURE_MAX = 0.4
            logger.warning(f"Using fallback temperature settings: MIN({self.TEMPERATURE_MIN}), BASE({self.TEMPERATURE_BASE}), MAX({self.TEMPERATURE_MAX})")
   
    def get_provider_config(self, model: str):
        """Get provider config for a specific model."""
        # Extract base model without suffixes like :free or :nitro
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

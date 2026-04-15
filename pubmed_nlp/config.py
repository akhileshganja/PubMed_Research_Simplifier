"""Configuration management for PubMed NLP pipeline."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # PubMed API
    pubmed_email: str = "tejaswinbadugu@gmail.com"
    pubmed_api_key: str | None = None
    
    # Database
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "pubmed_nlp"
    chroma_persist_dir: str = "./chroma_db"
    
    # Processing Limits
    max_articles_per_query: int = 100
    cache_enabled: bool = True
    
    # NLP Models
    scispacy_model: str = "en_core_sci_lg"
    ner_model_bc5cdr: str = "en_ner_bc5cdr_md"
    ner_model_jnlpba: str = "en_ner_jnlpba_md"
    sentence_transformer: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    summarization_model: str = "Falconsai/medical_summarization"
    
    # App Settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

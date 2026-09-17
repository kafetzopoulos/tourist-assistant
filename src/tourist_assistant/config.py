from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure API
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str = "2024-08-01-preview"

    # Weather / App
    weather_api_base: str = "https://api.open-meteo.com/v1"
    default_lat: float = 40.8475
    default_lon: float = 25.8744
    default_timezone: str = "Europe/Athens"

    # Date
    default_plan_start_hour: int = 9 

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    attractions_path: str = "src/tourist_assistant/data/attractions.json"
    faiss_index_path: str = ".cache/faiss.index"
    faiss_meta_path: str = ".cache/faiss_meta.json"

    # Conversation
    history_turns_to_llm: int = 6
    default_plan_hours: float = 4.0
    default_plan_start_offset_hours: int = 1
    weather_forecast_hours: int = 6

    # RAG retrieval
    rag_top_k_factual: int = 4
    rag_top_k_recommendation: int = 5
    rag_top_k_plan: int = 8
    rag_top_k_replacement: int = 5

    # Prompt guard
    prompt_guard_model: str = "skshreyas714/prompt-guard-finetuned"
    prompt_guard_threshold: float = 0.5
    prompt_guard_max_tokens: int = 512


settings = Settings()
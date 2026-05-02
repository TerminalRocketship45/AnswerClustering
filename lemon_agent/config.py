config = {
    "model": "gemini-2.0-flash",
    "criteria_sample_size": 20,
    "k_criteria": 5,
    "failure_modes_per_criterion": 2,
    "failure_mode_style": "statement",  # statement or question
    "batch_size": 10,
    "rating_levels": 3,
    "vector_mode": "failure_modes",
    "rag_enabled": False,
    "embeddings_model": "text-embedding-3-large",
    "max_llm_tokens": 1600,
    "retry_attempts": 1,
    "retry_backoff_seconds": 1.5,
    "rag_top_k": 3,
}

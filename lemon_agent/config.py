config = {
    "model": "gemini-2.5-flash",
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
    "architecture": "single",  # "single" or "multi"
    "search_enabled": False,  # enables RAG/search for agents
    "criteria_list": None,  # list of criteria for multi-agent, or None
    "fewshot_samples_path": None,  # path to JSON file for few-shot samples, or None
    "rationale": False,  # if True, require rationale in ratings
    "plot_mode": "criteria",  # "failure_modes" or "criteria"
}

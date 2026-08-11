"""Lightweight version definitions shared by packaging and runtime code."""

# .post2 marks the FinBase custom build (webui search + search param +
# smart_heading + reasoning-model fallback in llm/openai.py)
# .post3 adds: text ingestion (insert_text) resolves chunk strategy from
# LIGHTRAG_PARSER rules via file_source (aligned with file upload).
__version__ = "1.5.6.post3"
__api_version__ = "0328"

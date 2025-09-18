"""
RAG Engine for ATM Assist System

Complete Retrieval-Augmented Generation engine for intelligent
ATM troubleshooting and assistance. Combines vector search,
context retrieval, and specialized response generation.

Components:
- prompt_templates: ATM-specific response templates
- retriever: Context retrieval from hybrid vector store
- generator: Response generation using templates
- pipeline: Complete RAG processing pipeline

The RAG pipeline flow:
1. User Query → Query Embedding
2. Vector Search → Similar ATM Logs
3. Context Building → Relevant Information
4. Response Generation → Helpful Answer
"""

from .prompt_templates import PromptTemplates
from .retriever import ContextRetriever
from .generator import ResponseGenerator
from .pipeline import ATMRagPipeline

__all__ = [
    'PromptTemplates',
    'ContextRetriever',
    'ResponseGenerator',
    'ATMRagPipeline'
]
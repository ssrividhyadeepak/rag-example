"""
Database Schema Definitions

Defines the schema for storing ATM logs and vector metadata in MongoDB
with proper indexing for efficient queries and relationships.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import hashlib


@dataclass
class ATMLogSchema:
    """
    Schema for storing ATM log entries in MongoDB.

    This collection stores the original log data with rich metadata
    for filtering, aggregation, and analysis.
    """

    # Core log fields (required)
    log_id: str
    timestamp: datetime
    session_id: str
    operation: str
    status: str
    message: str

    # Optional fields
    customer_session_id: Optional[str] = None
    error_code: Optional[str] = None
    atm_id: Optional[str] = None
    amount: Optional[float] = None

    # Processed fields
    extracted_text: str = ""
    text_hash: str = ""

    # Metadata
    metadata: Dict[str, Any] = None

    # System fields
    created_at: datetime = None
    updated_at: datetime = None
    version: int = 1

    def __post_init__(self):
        """Initialize computed fields."""
        if self.metadata is None:
            self.metadata = {}

        if self.created_at is None:
            self.created_at = datetime.utcnow()

        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

        # Generate text hash for deduplication
        if self.extracted_text and not self.text_hash:
            self.text_hash = hashlib.sha256(self.extracted_text.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'customer_session_id': self.customer_session_id,
            'operation': self.operation,
            'status': self.status,
            'message': self.message,
            'error_code': self.error_code,
            'atm_id': self.atm_id,
            'amount': self.amount,
            'extracted_text': self.extracted_text,
            'text_hash': self.text_hash,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ATMLogSchema':
        """Create instance from MongoDB document."""
        return cls(
            log_id=data['log_id'],
            timestamp=data['timestamp'],
            session_id=data['session_id'],
            customer_session_id=data.get('customer_session_id'),
            operation=data['operation'],
            status=data['status'],
            message=data['message'],
            error_code=data.get('error_code'),
            atm_id=data.get('atm_id'),
            amount=data.get('amount'),
            extracted_text=data.get('extracted_text', ''),
            text_hash=data.get('text_hash', ''),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            version=data.get('version', 1)
        )


@dataclass
class VectorMetadataSchema:
    """
    Schema for storing vector metadata and FAISS index mappings.

    This collection links FAISS vector indices to MongoDB documents
    and stores embedding-related metadata.
    """

    # Link to ATM log
    log_id: str

    # FAISS index information
    faiss_index: int  # Position in FAISS index
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # Text information
    text_content: str = ""
    text_hash: str = ""
    text_length: int = 0

    # Vector metadata
    vector_norm: float = 0.0
    confidence_score: float = 1.0

    # Processing metadata
    processed_at: datetime = None
    model_version: str = "1.0"

    # System fields
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        """Initialize computed fields."""
        if self.processed_at is None:
            self.processed_at = datetime.utcnow()

        if self.created_at is None:
            self.created_at = datetime.utcnow()

        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

        # Calculate text length and hash
        if self.text_content:
            self.text_length = len(self.text_content)
            if not self.text_hash:
                self.text_hash = hashlib.sha256(self.text_content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return {
            'log_id': self.log_id,
            'faiss_index': self.faiss_index,
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions,
            'text_content': self.text_content,
            'text_hash': self.text_hash,
            'text_length': self.text_length,
            'vector_norm': self.vector_norm,
            'confidence_score': self.confidence_score,
            'processed_at': self.processed_at,
            'model_version': self.model_version,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorMetadataSchema':
        """Create instance from MongoDB document."""
        return cls(
            log_id=data['log_id'],
            faiss_index=data['faiss_index'],
            embedding_model=data.get('embedding_model', 'all-MiniLM-L6-v2'),
            embedding_dimensions=data.get('embedding_dimensions', 384),
            text_content=data.get('text_content', ''),
            text_hash=data.get('text_hash', ''),
            text_length=data.get('text_length', 0),
            vector_norm=data.get('vector_norm', 0.0),
            confidence_score=data.get('confidence_score', 1.0),
            processed_at=data.get('processed_at'),
            model_version=data.get('model_version', '1.0'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


class MongoDBIndexes:
    """
    Defines MongoDB indexes for optimal query performance.
    """

    ATM_LOGS_INDEXES = [
        # Primary queries
        [('log_id', 1)],  # Unique index
        [('timestamp', -1)],  # Time-based queries
        [('session_id', 1)],  # Session grouping

        # Filtering indexes
        [('operation', 1), ('status', 1)],  # Operation + status filtering
        [('atm_id', 1), ('timestamp', -1)],  # ATM-specific queries
        [('error_code', 1)],  # Error analysis

        # Compound indexes for complex queries
        [('operation', 1), ('status', 1), ('timestamp', -1)],  # Multi-field filtering
        [('metadata.is_error', 1), ('timestamp', -1)],  # Error log filtering

        # Text search
        [('text_hash', 1)],  # Deduplication
        [('extracted_text', 'text')]  # Full-text search
    ]

    VECTOR_METADATA_INDEXES = [
        # Primary queries
        [('log_id', 1)],  # Link to ATM logs
        [('faiss_index', 1)],  # FAISS position lookup

        # Vector metadata
        [('embedding_model', 1), ('embedding_dimensions', 1)],
        [('text_hash', 1)],  # Deduplication

        # Processing queries
        [('processed_at', -1)],  # Recent processing
        [('model_version', 1)]  # Model versioning
    ]

    @staticmethod
    def create_indexes(db, collection_name: str):
        """Create appropriate indexes for a collection."""
        collection = db[collection_name]

        if collection_name == 'atm_logs':
            indexes = MongoDBIndexes.ATM_LOGS_INDEXES
        elif collection_name == 'vector_metadata':
            indexes = MongoDBIndexes.VECTOR_METADATA_INDEXES
        else:
            return

        for index_spec in indexes:
            try:
                collection.create_index(index_spec)
            except Exception as e:
                print(f"Warning: Could not create index {index_spec}: {e}")


def get_sample_atm_log() -> ATMLogSchema:
    """Get a sample ATM log for testing."""
    return ATMLogSchema(
        log_id="log_sample_001",
        timestamp=datetime.utcnow(),
        session_id="SES_SAMPLE_001",
        customer_session_id="CUST_SAMPLE_001",
        operation="withdrawal",
        status="denied",
        message="Withdrawal denied DDL exceeded",
        error_code="DDL_EXCEEDED",
        atm_id="ATM001",
        amount=500.0,
        extracted_text="Time: 2024-01-15T10:30:00Z. Session: SES_SAMPLE_001. withdrawal denied. Withdrawal denied DDL exceeded. Error Code: DDL_EXCEEDED. ATM: ATM001. Amount: 500",
        metadata={
            "location": "Branch_Downtown",
            "card_number": "****1234",
            "is_error": True
        }
    )


def get_sample_vector_metadata() -> VectorMetadataSchema:
    """Get sample vector metadata for testing."""
    return VectorMetadataSchema(
        log_id="log_sample_001",
        faiss_index=0,
        text_content="Time: 2024-01-15T10:30:00Z. Session: SES_SAMPLE_001. withdrawal denied. Withdrawal denied DDL exceeded. Error Code: DDL_EXCEEDED. ATM: ATM001. Amount: 500",
        vector_norm=1.0,
        confidence_score=0.95
    )
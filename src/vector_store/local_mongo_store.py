"""
Local MongoDB Store

Handles all MongoDB operations for storing ATM logs and vector metadata
in a local MongoDB instance. Provides rich querying, filtering, and aggregation
capabilities for the ATM RAG system.
"""

from pymongo import MongoClient, errors
from pymongo.collection import Collection
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging

from .schema import ATMLogSchema, VectorMetadataSchema, MongoDBIndexes

logger = logging.getLogger(__name__)


class LocalMongoStore:
    """
    Local MongoDB operations for ATM logs and vector metadata.

    Provides a complete interface for storing, querying, and managing
    ATM log data with rich metadata support.
    """

    def __init__(self,
                 connection_string: str = "mongodb://localhost:27017",
                 database_name: str = "atm_rag",
                 timeout_ms: int = 5000):
        """
        Initialize MongoDB connection.

        Args:
            connection_string (str): MongoDB connection string
            database_name (str): Database name for ATM RAG data
            timeout_ms (int): Connection timeout in milliseconds
        """
        self.connection_string = connection_string
        self.database_name = database_name

        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=timeout_ms)
            # Test connection
            self.client.server_info()
            self.db = self.client[database_name]

            # Collections
            self.atm_logs: Collection = self.db.atm_logs
            self.vector_metadata: Collection = self.db.vector_metadata

            # Create indexes for performance
            self._create_indexes()

            logger.info(f"Connected to MongoDB at {connection_string}")

        except errors.ServerSelectionTimeoutError:
            logger.error(f"Could not connect to MongoDB at {connection_string}")
            logger.error("Make sure MongoDB is running locally:")
            logger.error("  brew install mongodb-community")
            logger.error("  brew services start mongodb-community")
            raise
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise

    def _create_indexes(self):
        """Create indexes for optimal query performance."""
        try:
            MongoDBIndexes.create_indexes(self.db, 'atm_logs')
            MongoDBIndexes.create_indexes(self.db, 'vector_metadata')
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create some indexes: {e}")

    # ATM Logs Operations

    def insert_atm_log(self, log_entry: ATMLogSchema) -> bool:
        """
        Insert a single ATM log entry.

        Args:
            log_entry (ATMLogSchema): Log entry to insert

        Returns:
            bool: True if inserted successfully
        """
        try:
            result = self.atm_logs.insert_one(log_entry.to_dict())
            logger.debug(f"Inserted ATM log: {log_entry.log_id}")
            return result.acknowledged
        except errors.DuplicateKeyError:
            logger.warning(f"ATM log already exists: {log_entry.log_id}")
            return False
        except Exception as e:
            logger.error(f"Error inserting ATM log {log_entry.log_id}: {e}")
            return False

    def insert_atm_logs_batch(self, log_entries: List[ATMLogSchema]) -> Dict[str, Any]:
        """
        Insert multiple ATM log entries efficiently.

        Args:
            log_entries (List[ATMLogSchema]): List of log entries

        Returns:
            Dict[str, Any]: Insert results summary
        """
        if not log_entries:
            return {"inserted": 0, "errors": 0, "duplicates": 0}

        documents = [log.to_dict() for log in log_entries]
        inserted = 0
        errors = 0
        duplicates = 0

        try:
            # Use ordered=False for better performance with duplicates
            result = self.atm_logs.insert_many(documents, ordered=False)
            inserted = len(result.inserted_ids)
        except errors.BulkWriteError as e:
            # Handle partial success with duplicates/errors
            inserted = e.details.get('nInserted', 0)
            for error in e.details.get('writeErrors', []):
                if error['code'] == 11000:  # Duplicate key
                    duplicates += 1
                else:
                    errors += 1
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            errors = len(log_entries)

        logger.info(f"Batch insert: {inserted} inserted, {duplicates} duplicates, {errors} errors")
        return {"inserted": inserted, "duplicates": duplicates, "errors": errors}

    def get_atm_log(self, log_id: str) -> Optional[ATMLogSchema]:
        """
        Retrieve a single ATM log by ID.

        Args:
            log_id (str): Log ID to retrieve

        Returns:
            Optional[ATMLogSchema]: Log entry or None if not found
        """
        try:
            doc = self.atm_logs.find_one({"log_id": log_id})
            if doc:
                return ATMLogSchema.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Error retrieving ATM log {log_id}: {e}")
            return None

    def query_atm_logs(self,
                      filters: Optional[Dict[str, Any]] = None,
                      sort_by: str = "timestamp",
                      sort_order: int = -1,
                      limit: int = 100,
                      skip: int = 0) -> List[ATMLogSchema]:
        """
        Query ATM logs with flexible filtering.

        Args:
            filters (Optional[Dict[str, Any]]): MongoDB query filters
            sort_by (str): Field to sort by
            sort_order (int): 1 for ascending, -1 for descending
            limit (int): Maximum number of results
            skip (int): Number of results to skip

        Returns:
            List[ATMLogSchema]: List of matching log entries
        """
        try:
            query = filters or {}
            cursor = self.atm_logs.find(query).sort(sort_by, sort_order).skip(skip).limit(limit)

            results = []
            for doc in cursor:
                try:
                    results.append(ATMLogSchema.from_dict(doc))
                except Exception as e:
                    logger.warning(f"Could not parse log document: {e}")
                    continue

            return results
        except Exception as e:
            logger.error(f"Error querying ATM logs: {e}")
            return []

    def filter_by_operation(self, operation: str, limit: int = 100) -> List[ATMLogSchema]:
        """Filter logs by operation type."""
        return self.query_atm_logs(
            filters={"operation": operation},
            limit=limit
        )

    def filter_by_status(self, status: str, limit: int = 100) -> List[ATMLogSchema]:
        """Filter logs by status."""
        return self.query_atm_logs(
            filters={"status": status},
            limit=limit
        )

    def filter_by_error_code(self, error_code: str, limit: int = 100) -> List[ATMLogSchema]:
        """Filter logs by error code."""
        return self.query_atm_logs(
            filters={"error_code": error_code},
            limit=limit
        )

    def filter_by_atm_id(self, atm_id: str, limit: int = 100) -> List[ATMLogSchema]:
        """Filter logs by ATM ID."""
        return self.query_atm_logs(
            filters={"atm_id": atm_id},
            limit=limit
        )

    def filter_by_date_range(self,
                           start_date: datetime,
                           end_date: datetime,
                           limit: int = 100) -> List[ATMLogSchema]:
        """Filter logs by date range."""
        return self.query_atm_logs(
            filters={
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            },
            limit=limit
        )

    def get_error_logs_only(self, limit: int = 100) -> List[ATMLogSchema]:
        """Get only error logs (failed, denied, error status)."""
        return self.query_atm_logs(
            filters={
                "$or": [
                    {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                    {"error_code": {"$exists": True, "$ne": ""}},
                    {"metadata.is_error": True}
                ]
            },
            limit=limit
        )

    # Vector Metadata Operations

    def insert_vector_metadata(self, metadata: VectorMetadataSchema) -> bool:
        """Insert vector metadata entry."""
        try:
            result = self.vector_metadata.insert_one(metadata.to_dict())
            logger.debug(f"Inserted vector metadata for log: {metadata.log_id}")
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error inserting vector metadata {metadata.log_id}: {e}")
            return False

    def insert_vector_metadata_batch(self, metadata_list: List[VectorMetadataSchema]) -> Dict[str, Any]:
        """Insert multiple vector metadata entries."""
        if not metadata_list:
            return {"inserted": 0, "errors": 0}

        documents = [meta.to_dict() for meta in metadata_list]

        try:
            result = self.vector_metadata.insert_many(documents, ordered=False)
            inserted = len(result.inserted_ids)
            logger.info(f"Inserted {inserted} vector metadata entries")
            return {"inserted": inserted, "errors": 0}
        except Exception as e:
            logger.error(f"Batch vector metadata insert error: {e}")
            return {"inserted": 0, "errors": len(metadata_list)}

    def get_vector_metadata_by_log_id(self, log_id: str) -> Optional[VectorMetadataSchema]:
        """Get vector metadata by log ID."""
        try:
            doc = self.vector_metadata.find_one({"log_id": log_id})
            if doc:
                return VectorMetadataSchema.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector metadata for {log_id}: {e}")
            return None

    def get_vector_metadata_by_faiss_index(self, faiss_index: int) -> Optional[VectorMetadataSchema]:
        """Get vector metadata by FAISS index position."""
        try:
            doc = self.vector_metadata.find_one({"faiss_index": faiss_index})
            if doc:
                return VectorMetadataSchema.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector metadata for FAISS index {faiss_index}: {e}")
            return None

    def get_all_vector_metadata(self, limit: int = 10000) -> List[VectorMetadataSchema]:
        """Get all vector metadata entries."""
        try:
            cursor = self.vector_metadata.find().limit(limit)
            results = []
            for doc in cursor:
                try:
                    results.append(VectorMetadataSchema.from_dict(doc))
                except Exception as e:
                    logger.warning(f"Could not parse vector metadata: {e}")
                    continue
            return results
        except Exception as e:
            logger.error(f"Error retrieving vector metadata: {e}")
            return []

    # Analytics and Statistics

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored logs."""
        try:
            total_logs = self.atm_logs.count_documents({})

            # Operation statistics
            operation_stats = list(self.atm_logs.aggregate([
                {"$group": {"_id": "$operation", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))

            # Status statistics
            status_stats = list(self.atm_logs.aggregate([
                {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))

            # Error code statistics
            error_stats = list(self.atm_logs.aggregate([
                {"$match": {"error_code": {"$exists": True, "$ne": ""}}},
                {"$group": {"_id": "$error_code", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))

            # ATM statistics
            atm_stats = list(self.atm_logs.aggregate([
                {"$match": {"atm_id": {"$exists": True, "$ne": ""}}},
                {"$group": {"_id": "$atm_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))

            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_count = self.atm_logs.count_documents({
                "timestamp": {"$gte": yesterday}
            })

            return {
                "total_logs": total_logs,
                "recent_logs_24h": recent_count,
                "operations": {stat["_id"]: stat["count"] for stat in operation_stats},
                "statuses": {stat["_id"]: stat["count"] for stat in status_stats},
                "error_codes": {stat["_id"]: stat["count"] for stat in error_stats},
                "atm_distribution": {stat["_id"]: stat["count"] for stat in atm_stats}
            }
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return {}

    def get_vector_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored vectors."""
        try:
            total_vectors = self.vector_metadata.count_documents({})

            # Model statistics
            model_stats = list(self.vector_metadata.aggregate([
                {"$group": {"_id": "$embedding_model", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))

            # Average statistics
            avg_stats = list(self.vector_metadata.aggregate([
                {"$group": {
                    "_id": None,
                    "avg_text_length": {"$avg": "$text_length"},
                    "avg_vector_norm": {"$avg": "$vector_norm"},
                    "avg_confidence": {"$avg": "$confidence_score"}
                }}
            ]))

            avg_data = avg_stats[0] if avg_stats else {}

            return {
                "total_vectors": total_vectors,
                "embedding_models": {stat["_id"]: stat["count"] for stat in model_stats},
                "avg_text_length": avg_data.get("avg_text_length", 0),
                "avg_vector_norm": avg_data.get("avg_vector_norm", 0),
                "avg_confidence": avg_data.get("avg_confidence", 0)
            }
        except Exception as e:
            logger.error(f"Error getting vector statistics: {e}")
            return {}

    # Utility Methods

    def health_check(self) -> Dict[str, Any]:
        """Check MongoDB connection and database health."""
        try:
            # Test connection
            self.client.server_info()

            # Test collections
            log_count = self.atm_logs.count_documents({})
            vector_count = self.vector_metadata.count_documents({})

            return {
                "status": "healthy",
                "connection": "connected",
                "database": self.database_name,
                "atm_logs_count": log_count,
                "vector_metadata_count": vector_count,
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }

    def clear_all_data(self, confirm: str = None) -> bool:
        """
        Clear all data from collections. USE WITH CAUTION!

        Args:
            confirm (str): Must be "CONFIRM_DELETE_ALL_DATA" to proceed

        Returns:
            bool: True if cleared successfully
        """
        if confirm != "CONFIRM_DELETE_ALL_DATA":
            logger.warning("Clear all data operation requires confirmation")
            return False

        try:
            logs_result = self.atm_logs.delete_many({})
            vectors_result = self.vector_metadata.delete_many({})

            logger.warning(f"Cleared {logs_result.deleted_count} logs and {vectors_result.deleted_count} vector metadata entries")
            return True
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
#!/usr/bin/env python3
"""
Migration Script: Component 2 → Component 3

Migrates cached embedding data from Component 2 (Local Embeddings Generator)
to Component 3 (Vector Store & RAG) by loading existing embeddings and
ATM logs into the hybrid vector store (MongoDB + FAISS).

This script ensures continuity of data when transitioning between components.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.log_processor import LogReader, LogParser, TextExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import HybridVectorStore
from src.vector_store.schema import ATMLogSchema, VectorMetadataSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Component2Migrator:
    """
    Migrates data from Component 2 (embeddings) to Component 3 (vector store).

    Handles loading existing cached embeddings and ATM logs,
    then inserting them into the hybrid vector store.
    """

    def __init__(self,
                 logs_directory: str = "data/logs",
                 embeddings_cache: str = "data/embeddings/cache",
                 mongodb_uri: str = "mongodb://localhost:27017",
                 vector_store_path: str = "data/vector_store"):
        """
        Initialize migrator with source and destination paths.

        Args:
            logs_directory: Directory containing ATM log files
            embeddings_cache: Directory with cached embeddings
            mongodb_uri: MongoDB connection string
            vector_store_path: Vector store data directory
        """
        self.logs_directory = Path(logs_directory)
        self.embeddings_cache = Path(embeddings_cache)
        self.mongodb_uri = mongodb_uri
        self.vector_store_path = vector_store_path

        # Initialize components
        self.log_reader = LogReader(str(self.logs_directory))
        self.log_parser = LogParser()
        self.text_extractor = TextExtractor()
        self.embedding_generator = EmbeddingGenerator(
            cache_dir=str(self.embeddings_cache)
        )

        # Initialize vector store
        self.vector_store = HybridVectorStore(
            mongodb_uri=self.mongodb_uri,
            storage_path=self.vector_store_path
        )

        # Migration statistics
        self.stats = {
            "logs_processed": 0,
            "logs_migrated": 0,
            "embeddings_found": 0,
            "embeddings_migrated": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info("Component 2 → 3 Migrator initialized")

    def migrate_all_data(self) -> Dict[str, Any]:
        """
        Migrate all data from Component 2 to Component 3.

        Returns:
            Migration results and statistics
        """
        self.stats["start_time"] = datetime.utcnow()
        logger.info("Starting Component 2 → 3 data migration")

        try:
            # Step 1: Load all ATM logs
            atm_logs = self._load_atm_logs()
            logger.info(f"Loaded {len(atm_logs)} ATM logs")

            # Step 2: Load existing embeddings
            embeddings_data = self._load_existing_embeddings(atm_logs)
            logger.info(f"Found embeddings for {len(embeddings_data)} logs")

            # Step 3: Generate missing embeddings
            complete_embeddings = self._ensure_complete_embeddings(atm_logs, embeddings_data)
            logger.info(f"Total embeddings ready: {len(complete_embeddings)}")

            # Step 4: Migrate to vector store
            migration_result = self._migrate_to_vector_store(atm_logs, complete_embeddings)
            logger.info("Migration to vector store completed")

            # Step 5: Verify migration
            verification_result = self._verify_migration()
            logger.info("Migration verification completed")

            self.stats["end_time"] = datetime.utcnow()
            total_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            final_result = {
                "success": True,
                "migration_stats": self.stats,
                "migration_results": migration_result,
                "verification": verification_result,
                "total_time_seconds": total_time
            }

            logger.info(f"Migration completed successfully in {total_time:.2f} seconds")
            return final_result

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Migration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "migration_stats": self.stats
            }

    def _load_atm_logs(self) -> List[ATMLogSchema]:
        """
        Load all ATM logs from the logs directory.

        Returns:
            List of ATMLogSchema objects
        """
        atm_logs = []

        try:
            # Get all available log files
            log_files = self.log_reader.get_available_files()
            logger.info(f"Found {len(log_files)} log files to process")

            for log_file in log_files:
                try:
                    # Read log entries
                    log_entries = self.log_reader.read_log_file(log_file)
                    logger.debug(f"Read {len(log_entries)} entries from {log_file}")

                    # Parse each log entry
                    for entry in log_entries:
                        try:
                            parsed_log = self.log_parser.parse_log_entry(entry)
                            extracted_text = self.text_extractor.extract_searchable_text(parsed_log)

                            # Create ATMLogSchema
                            atm_log = ATMLogSchema(
                                log_id=parsed_log["log_id"],
                                timestamp=parsed_log["timestamp"],
                                session_id=parsed_log["session_id"],
                                customer_session_id=parsed_log.get("customer_session_id"),
                                operation=parsed_log["operation"],
                                status=parsed_log["status"],
                                message=parsed_log["message"],
                                error_code=parsed_log.get("error_code"),
                                atm_id=parsed_log.get("atm_id"),
                                amount=parsed_log.get("amount"),
                                extracted_text=extracted_text,
                                metadata=parsed_log.get("metadata", {})
                            )

                            atm_logs.append(atm_log)
                            self.stats["logs_processed"] += 1

                        except Exception as e:
                            logger.warning(f"Error parsing log entry: {e}")
                            self.stats["errors"] += 1

                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
                    self.stats["errors"] += 1

            return atm_logs

        except Exception as e:
            logger.error(f"Error loading ATM logs: {e}")
            raise

    def _load_existing_embeddings(self, atm_logs: List[ATMLogSchema]) -> Dict[str, np.ndarray]:
        """
        Load existing embeddings from Component 2 cache.

        Args:
            atm_logs: List of ATM logs to check for embeddings

        Returns:
            Dictionary mapping log_id to embedding vector
        """
        embeddings_data = {}

        try:
            for atm_log in atm_logs:
                # Try to get cached embedding
                embedding = self.embedding_generator.get_cached_embedding(atm_log.extracted_text)

                if embedding is not None:
                    embeddings_data[atm_log.log_id] = embedding
                    self.stats["embeddings_found"] += 1

            logger.info(f"Loaded {len(embeddings_data)} cached embeddings")
            return embeddings_data

        except Exception as e:
            logger.error(f"Error loading existing embeddings: {e}")
            return {}

    def _ensure_complete_embeddings(self,
                                   atm_logs: List[ATMLogSchema],
                                   existing_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ensure all logs have embeddings, generating missing ones.

        Args:
            atm_logs: List of ATM logs
            existing_embeddings: Already available embeddings

        Returns:
            Complete embeddings dictionary
        """
        complete_embeddings = existing_embeddings.copy()
        missing_logs = []

        # Find logs without embeddings
        for atm_log in atm_logs:
            if atm_log.log_id not in complete_embeddings:
                missing_logs.append(atm_log)

        if missing_logs:
            logger.info(f"Generating embeddings for {len(missing_logs)} missing logs")

            # Extract texts for batch processing
            texts = [log.extracted_text for log in missing_logs]
            log_ids = [log.log_id for log in missing_logs]

            try:
                # Generate embeddings in batch
                new_embeddings = self.embedding_generator.generate_embeddings(texts)

                # Add to complete embeddings
                for log_id, embedding in zip(log_ids, new_embeddings):
                    complete_embeddings[log_id] = embedding

                logger.info(f"Generated {len(new_embeddings)} new embeddings")

            except Exception as e:
                logger.error(f"Error generating missing embeddings: {e}")
                # Continue with partial data
                self.stats["errors"] += 1

        return complete_embeddings

    def _migrate_to_vector_store(self,
                                atm_logs: List[ATMLogSchema],
                                embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Migrate ATM logs and embeddings to the vector store.

        Args:
            atm_logs: List of ATM logs
            embeddings: Embeddings dictionary

        Returns:
            Migration results
        """
        logger.info("Starting migration to vector store")

        try:
            # Prepare data for batch insertion
            logs_to_insert = []
            embeddings_to_insert = []
            log_ids_to_insert = []

            for atm_log in atm_logs:
                if atm_log.log_id in embeddings:
                    logs_to_insert.append(atm_log)
                    embeddings_to_insert.append(embeddings[atm_log.log_id])
                    log_ids_to_insert.append(atm_log.log_id)

            if not logs_to_insert:
                logger.warning("No complete log-embedding pairs to migrate")
                return {"logs_inserted": 0, "vectors_inserted": 0}

            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings_to_insert)

            logger.info(f"Migrating {len(logs_to_insert)} log-embedding pairs")

            # Insert into vector store
            result = self.vector_store.insert_log_embeddings(
                log_schemas=logs_to_insert,
                embeddings=embeddings_array
            )

            self.stats["logs_migrated"] = result.get("mongodb_inserted", 0)
            self.stats["embeddings_migrated"] = result.get("faiss_added", 0)

            logger.info(f"Migration results: {result}")
            return result

        except Exception as e:
            logger.error(f"Error migrating to vector store: {e}")
            self.stats["errors"] += 1
            raise

    def _verify_migration(self) -> Dict[str, Any]:
        """
        Verify the migration was successful.

        Returns:
            Verification results
        """
        logger.info("Verifying migration")

        try:
            # Get vector store statistics
            vector_store_stats = self.vector_store.get_statistics()

            # Perform health check
            health_check = self.vector_store.health_check()

            # Test a simple search
            test_query_embedding = self.embedding_generator.generate_embedding("ATM withdrawal test")
            search_results = self.vector_store.search_similar_logs(
                query_vector=test_query_embedding,
                top_k=5
            )

            verification_result = {
                "vector_store_stats": vector_store_stats,
                "health_check": health_check,
                "test_search_results": len(search_results),
                "verification_passed": (
                    health_check.get("status") == "healthy" and
                    vector_store_stats.get("mongodb_docs", 0) > 0 and
                    vector_store_stats.get("faiss_vectors", 0) > 0
                )
            }

            if verification_result["verification_passed"]:
                logger.info("Migration verification PASSED")
            else:
                logger.warning("Migration verification FAILED")

            return verification_result

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {
                "verification_passed": False,
                "error": str(e)
            }

    def cleanup_component2_cache(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clean up Component 2 cache files after successful migration.

        Args:
            confirm: Must be True to actually delete files

        Returns:
            Cleanup results
        """
        if not confirm:
            return {
                "cleanup_performed": False,
                "message": "Cleanup requires explicit confirmation"
            }

        try:
            import shutil

            if self.embeddings_cache.exists():
                # Move cache to backup location
                backup_path = self.embeddings_cache.parent / f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(self.embeddings_cache), str(backup_path))

                logger.info(f"Component 2 cache moved to backup: {backup_path}")
                return {
                    "cleanup_performed": True,
                    "backup_location": str(backup_path)
                }
            else:
                return {
                    "cleanup_performed": False,
                    "message": "Cache directory not found"
                }

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                "cleanup_performed": False,
                "error": str(e)
            }

    def get_migration_summary(self) -> str:
        """Get a human-readable migration summary."""
        if not self.stats["start_time"]:
            return "Migration has not been started yet."

        duration = ""
        if self.stats["end_time"]:
            total_seconds = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            duration = f"Duration: {total_seconds:.2f} seconds"

        summary = f"""
Component 2 → 3 Migration Summary
================================
Logs processed: {self.stats['logs_processed']}
Logs migrated: {self.stats['logs_migrated']}
Embeddings found: {self.stats['embeddings_found']}
Embeddings migrated: {self.stats['embeddings_migrated']}
Errors encountered: {self.stats['errors']}
{duration}

Status: {'✓ COMPLETED' if self.stats['end_time'] else '⏳ IN PROGRESS'}
"""
        return summary.strip()


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Component 2 data to Component 3 vector store"
    )
    parser.add_argument(
        "--logs-dir",
        default="data/logs",
        help="Directory containing ATM log files"
    )
    parser.add_argument(
        "--embeddings-cache",
        default="data/embeddings/cache",
        help="Component 2 embeddings cache directory"
    )
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection string"
    )
    parser.add_argument(
        "--vector-store-path",
        default="data/vector_store",
        help="Vector store data directory"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up Component 2 cache after successful migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")

    # Initialize migrator
    migrator = Component2Migrator(
        logs_directory=args.logs_dir,
        embeddings_cache=args.embeddings_cache,
        mongodb_uri=args.mongodb_uri,
        vector_store_path=args.vector_store_path
    )

    try:
        if args.dry_run:
            # Just show what would be migrated
            print("\nDry run results:")
            print("- Would load ATM logs from:", args.logs_dir)
            print("- Would check embeddings cache:", args.embeddings_cache)
            print("- Would migrate to MongoDB:", args.mongodb_uri)
            print("- Would create vector store at:", args.vector_store_path)
            return

        # Perform actual migration
        print("Starting Component 2 → 3 migration...")
        result = migrator.migrate_all_data()

        # Print results
        print("\n" + migrator.get_migration_summary())

        if result["success"]:
            print("\n✓ Migration completed successfully!")

            # Clean up if requested
            if args.cleanup:
                print("\nCleaning up Component 2 cache...")
                cleanup_result = migrator.cleanup_component2_cache(confirm=True)
                if cleanup_result["cleanup_performed"]:
                    print(f"✓ Cache backed up to: {cleanup_result['backup_location']}")
                else:
                    print(f"⚠ Cleanup failed: {cleanup_result.get('message', 'Unknown error')}")

        else:
            print(f"\n✗ Migration failed: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        print(f"\n✗ Migration script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
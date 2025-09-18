"""
JSON Log Reader Module

Handles reading and loading JSON log files with proper error handling.
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogReader:
    """
    Reads and loads ATM log files from JSON format.
    """

    def __init__(self, log_directory: str = "data/logs"):
        """
        Initialize the LogReader.

        Args:
            log_directory (str): Directory containing log files
        """
        self.log_directory = log_directory
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure the log directory exists."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            logger.info(f"Created log directory: {self.log_directory}")

    def read_log_file(self, filename: str) -> List[Dict[str, Any]]:
        """
        Read a single JSON log file.

        Args:
            filename (str): Name of the log file to read

        Returns:
            List[Dict[str, Any]]: List of log entries

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        file_path = os.path.join(self.log_directory, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError(f"Expected list or dict, got {type(data)}")

            logger.info(f"Successfully read {len(data)} log entries from {filename}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise

    def read_all_logs(self, file_pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Read all JSON log files in the directory.

        Args:
            file_pattern (str): Pattern to match files (default: "*.json")

        Returns:
            List[Dict[str, Any]]: Combined list of all log entries
        """
        import glob

        pattern = os.path.join(self.log_directory, file_pattern)
        log_files = glob.glob(pattern)

        if not log_files:
            logger.warning(f"No log files found matching pattern: {pattern}")
            return []

        all_logs = []
        for file_path in log_files:
            filename = os.path.basename(file_path)
            try:
                logs = self.read_log_file(filename)
                all_logs.extend(logs)
            except Exception as e:
                logger.error(f"Skipping file {filename} due to error: {e}")
                continue

        logger.info(f"Read total of {len(all_logs)} log entries from {len(log_files)} files")
        return all_logs

    def get_available_files(self) -> List[str]:
        """
        Get list of available JSON log files.

        Returns:
            List[str]: List of available log file names
        """
        if not os.path.exists(self.log_directory):
            return []

        files = [f for f in os.listdir(self.log_directory)
                if f.endswith('.json') and os.path.isfile(os.path.join(self.log_directory, f))]

        return sorted(files)
#!/usr/bin/env python
"""Utility script to fix JSON serialization issues with NumPy datatypes."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def fix_json_file(filepath):
    """Fix a JSON file with NumPy serialization issues.

    Args:
        filepath: Path to the JSON file to fix

    Returns:
        bool: True if successful, False otherwise
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return False

    try:
        # Try to read the file as text
        with open(filepath, "r") as f:
            content = f.read()

        # Try to manually convert problematic NumPy types
        content = content.replace("NaN", "null")

        # Manual replacements for common NumPy types
        for np_type in [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
        ]:
            content = content.replace(f'"{np_type}(', '"')
            content = content.replace(f')"', '"')

        # Try to parse and rewrite the file
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try direct conversion
            logger.info("Direct parsing failed, trying to fix the file manually")

            # Create a backup
            backup_path = filepath.with_suffix(".json.bak")
            with open(backup_path, "w") as f:
                f.write(content)
            logger.info(f"Backup created at {backup_path}")

            # Try to fix with regex
            import re

            # Pattern to find numpy types like "int64(123)"
            np_pattern = r'"([a-z]+\d+)\(([^)]+)\)"'

            # Replace with just the value
            content = re.sub(np_pattern, r'"\2"', content)

            # Try parsing again
            data = json.loads(content)

        # Write back with the proper encoder
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Successfully fixed JSON file: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to fix JSON file: {e}")
        return False


def main():
    """Run the JSON fixer CLI."""
    parser = argparse.ArgumentParser(
        description="Fix JSON files with NumPy serialization issues"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the JSON file to fix",
    )

    args = parser.parse_args()

    return 0 if fix_json_file(args.filepath) else 1


if __name__ == "__main__":
    exit(main())

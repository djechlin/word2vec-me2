#!/usr/bin/env python3
"""
CLI script to download Amazon Reviews 2023 dataset from Hugging Face.

Usage:
    python download_amazon_reviews.py --category "Books" --output_dir "./data" --type "reviews"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from lib module
from datasets import load_dataset

from lib.amazon_data import CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_amazon_reviews(
    category: str,
    data_type: Literal["reviews", "metadata"],
    output_dir: str,
    sample_size: int | None = None,
) -> str:
    """
    Download Amazon Reviews dataset for a specific category.

    Args:
        category: Product category to download (e.g., "Books")
        data_type: Type of data to download ("reviews" or "metadata")
        output_dir: Directory to save the downloaded data
        sample_size: Optional number of samples to keep (for testing with smaller datasets)

    Returns:
        Path to the saved dataset
    """
    # Validate category
    if category not in CATEGORIES:
        logger.warning(f"Category '{category}' not in known list. Trying anyway...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset configuration based on type
    if data_type == "reviews":
        config = f"raw_review_{category}"
    else:
        config = f"raw_meta_{category}"

    logger.info(f"Downloading {data_type} for {category} category...")

    try:
        # Load dataset from Hugging Face
        if data_type == "metadata":
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", config, split="full", trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", config, trust_remote_code=True
            )

        # Take a sample if requested
        if sample_size and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))
            logger.info(f"Selected sample of {sample_size} records")

        # Save dataset to disk
        output_path = os.path.join(output_dir, f"{category}_{data_type}.arrow")
        dataset.save_to_disk(output_path)
        logger.info(f"Dataset saved to {output_path}")

        # Log some information about the dataset
        logger.info(f"Dataset size: {len(dataset)} records")
        logger.info(f"Dataset features: {dataset.column_names}")

        return output_path

    except Exception as e:
        logger.error(f"Error downloading dataset: {e!s}")
        raise


def list_available_categories() -> None:
    """Print all available categories in the dataset."""
    logger.info("Available categories in Amazon Reviews 2023 dataset:")
    for category in CATEGORIES:
        print(f"- {category}")


def main() -> int:
    """Main function to parse arguments and download dataset."""
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument(
        "--category",
        type=str,
        default="Books",
        help="Category to download (e.g., Books, Electronics)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Directory to save the dataset"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["reviews", "metadata"],
        default="reviews",
        help="Type of data to download (reviews or metadata)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Number of samples to keep (for testing)"
    )
    parser.add_argument(
        "--list_categories", action="store_true", help="List all available categories and exit"
    )

    args = parser.parse_args()

    # List categories if requested
    if args.list_categories:
        list_available_categories()
        return 0

    # Download the dataset
    try:
        output_path = download_amazon_reviews(
            category=args.category,
            data_type=args.type,
            output_dir=args.output_dir,
            sample_size=args.sample,
        )
        logger.info(f"Successfully downloaded dataset to {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to download dataset: {e!s}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

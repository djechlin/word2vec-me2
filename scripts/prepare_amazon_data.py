#!/usr/bin/env python3
"""
CLI script to prepare Amazon Reviews dataset for Word2Vec training.
Converts the Arrow format dataset to a text corpus suitable for the main.py script.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from lib
from lib.amazon_data import convert_amazon_reviews_to_corpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews for Word2Vec training")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the Arrow dataset directory (e.g., data/Books_reviews.arrow)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the text corpus (e.g., data/amazon_books_corpus.txt)')
    parser.add_argument('--max_reviews', type=int, default=None,
                        help='Maximum number of reviews to include')
    parser.add_argument('--min_length', type=int, default=20,
                        help='Minimum review length in characters')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        # Process the dataset
        convert_amazon_reviews_to_corpus(
            input_path=args.input,
            output_path=args.output,
            max_reviews=args.max_reviews,
            min_length=args.min_length
        )
        logger.info(f"Text corpus successfully created at {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
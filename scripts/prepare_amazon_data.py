#!/usr/bin/env python3
"""
Script to prepare Amazon Reviews dataset for Word2Vec training.
Converts the Arrow format dataset to a text corpus suitable for the main.py script.
"""

import argparse
import logging
import os
from datasets import load_from_disk
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_amazon_reviews(
    input_path: str,
    output_path: str,
    max_reviews: Optional[int] = None,
    min_length: int = 20  # Minimum review length in characters
) -> None:
    """
    Convert Amazon Reviews dataset to a text corpus for Word2Vec training.
    
    Args:
        input_path: Path to the Arrow dataset directory
        output_path: Path to save the text corpus
        max_reviews: Maximum number of reviews to include (for testing)
        min_length: Minimum review length in characters
    """
    logger.info(f"Loading Amazon Reviews dataset from {input_path}")
    
    # Load the dataset
    try:
        dataset = load_from_disk(input_path)
        logger.info(f"Dataset loaded successfully with {len(dataset)} reviews")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    # Limit dataset size if specified
    if max_reviews and max_reviews < len(dataset):
        dataset = dataset.select(range(max_reviews))
        logger.info(f"Limited dataset to {max_reviews} reviews")
    
    # Extract and clean review texts
    logger.info("Extracting review texts...")
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        processed_count = 0
        skipped_count = 0
        
        for i, review in enumerate(dataset):
            # Amazon Reviews dataset has 'review_text' column
            review_text = review.get('review_text', '')
            
            # Skip short reviews
            if len(review_text) < min_length:
                skipped_count += 1
                continue
            
            # Clean the text (remove newlines to match main.py's expected format)
            clean_text = review_text.replace('\n', ' ').strip()
            
            # Write to corpus file - each review as separate "article" with blank line between
            out_file.write(clean_text + '\n\n')
            
            processed_count += 1
            if processed_count % 10000 == 0:
                logger.info(f"Processed {processed_count} reviews")
    
    logger.info(f"Finished processing. Included {processed_count} reviews, skipped {skipped_count} reviews.")
    logger.info(f"Corpus saved to {output_path}")

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
    
    # Process the dataset
    prepare_amazon_reviews(
        input_path=args.input,
        output_path=args.output,
        max_reviews=args.max_reviews,
        min_length=args.min_length
    )

if __name__ == "__main__":
    main()
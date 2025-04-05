#!/usr/bin/env python3
"""
CLI script to run the complete Amazon Reviews Word2Vec pipeline:
1. Prepare Amazon data from Arrow format to text corpus
2. Train Word2Vec model using library functions
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from library modules
from lib.amazon_data import convert_amazon_reviews_to_corpus
from lib.word2vec import (
    tokenize_with_nltk,
    tokenize_with_tiktoken,
    process_corpus,
    train_word2vec,
    evaluate_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s'
)
logger = logging.getLogger(__name__)

def run_amazon_word2vec_pipeline(
    category: str,
    max_reviews: int,
    tokenizer_name: str,
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    epochs: int,
    output_dir: str,
) -> int:
    """
    Run the complete Amazon Reviews Word2Vec pipeline.

    Args:
        category: Amazon category (e.g., Books)
        max_reviews: Maximum number of reviews to process
        tokenizer_name: Tokenizer to use ('nltk' or 'tiktoken')
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum count of words to consider
        sg: Training algorithm (1 for skip-gram, 0 for CBOW)
        epochs: Number of training epochs
        output_dir: Directory to save models

    Returns:
        0 on success, error code otherwise
    """
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / f'{category}_reviews.arrow'
    corpus_path = project_root / 'data' / f'{category.lower()}_corpus.txt'
    output_prefix = f"{output_dir}/{category.lower()}_word2vec"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert Amazon data to text corpus
    logger.info(f"Step 1: Converting Amazon {category} reviews to text corpus...")
    try:
        convert_amazon_reviews_to_corpus(
            input_path=str(input_path),
            output_path=str(corpus_path),
            max_reviews=max_reviews
        )
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        return 1

    # Step 2: Process and train Word2Vec model
    logger.info(f"Step 2: Training Word2Vec model with {tokenizer_name} tokenizer...")

    # Select tokenizer function
    tokenizer = tokenize_with_nltk if tokenizer_name == 'nltk' else tokenize_with_tiktoken

    try:
        # Process corpus
        tokenized_corpus = process_corpus(
            corpus_path=str(corpus_path),
            tokenizer=tokenizer
        )

        if not tokenized_corpus:
            logger.error("No valid articles found in corpus. Exiting.")
            return 1

        # Train model
        model = train_word2vec(
            tokenized_corpus=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs
        )

        # Basic evaluation
        word_pairs: List[Tuple[str, str]] = [
            ('man', 'woman'),
            ('king', 'queen'),
            ('computer', 'laptop'),
            ('good', 'bad'),
            ('cat', 'dog')
        ]
        evaluate_model(model, word_pairs)

        # Save model
        output_path = f"{output_prefix}_{tokenizer_name}.model"
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")

        # Save word vectors for easier use
        vector_path = f"{output_prefix}_{tokenizer_name}.wordvectors"
        model.wv.save(vector_path)
        logger.info(f"Word vectors saved to {vector_path}")

    except Exception as e:
        logger.error(f"Word2Vec training failed: {str(e)}")
        return 1

    logger.info("Amazon Reviews Word2Vec pipeline completed successfully!")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Run Amazon Reviews Word2Vec Pipeline")
    parser.add_argument('--category', type=str, default='Books',
                        help='Amazon category to process (e.g., Books, Electronics)')
    parser.add_argument('--max_reviews', type=int, default=100000,
                        help='Maximum number of reviews to include (default: 100000)')
    parser.add_argument('--tokenizer', type=str, choices=['nltk', 'tiktoken'], default='nltk',
                        help='Tokenizer to use: nltk or tiktoken')
    parser.add_argument('--vector_size', type=int, default=100,
                        help='Dimensionality of the word vectors')
    parser.add_argument('--window', type=int, default=5,
                        help='Maximum distance between current and predicted word')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1],
                        help='Training algorithm: 1 for skip-gram, 0 for CBOW')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum count of words to consider for training')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save the output models')

    args = parser.parse_args()

    return run_amazon_word2vec_pipeline(
        category=args.category,
        max_reviews=args.max_reviews,
        tokenizer_name=args.tokenizer,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    sys.exit(main())
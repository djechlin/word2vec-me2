"""
Library for handling Amazon Reviews data processing
"""

import logging
import os
from datasets import load_from_disk
from typing import Optional, List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Available categories in the dataset
CATEGORIES = [
    "All_Beauty", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Luxury_Beauty",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Prime_Pantry", "Software", "Sports_and_Outdoors",
    "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"
]

def convert_amazon_reviews_to_corpus(
    input_path: str,
    output_path: str,
    max_reviews: Optional[int] = None,
    min_length: int = 20  # Minimum review length in characters
) -> None:
    """
    Convert Amazon Reviews dataset from Arrow format to a plain text corpus for Word2Vec training.

    This function:
    1. Loads the Amazon reviews dataset from the Arrow format
    2. Filters out reviews shorter than min_length
    3. Cleans the text (removes newlines, extra spaces)
    4. Writes each review as a separate "article" with blank lines between them
    5. Limits the number of reviews if max_reviews is specified

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

def get_review_statistics(
    input_path: str,
    max_reviews: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get statistics from Amazon reviews dataset

    Args:
        input_path: Path to the Arrow dataset directory
        max_reviews: Maximum number of reviews to analyze

    Returns:
        Dictionary with statistics about the dataset
    """
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

    total_reviews = len(dataset)
    total_words = 0
    ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    review_lengths = []

    for review in dataset:
        # Count words in each review
        review_text = review.get('review_text', '')
        words = review_text.split()
        review_lengths.append(len(words))
        total_words += len(words)

        # Count rating distribution
        rating = review.get('rating', 0)
        if rating in ratings:
            ratings[rating] += 1

    # Calculate statistics
    avg_review_length = total_words / total_reviews if total_reviews > 0 else 0

    return {
        "total_reviews": total_reviews,
        "total_words": total_words,
        "average_review_length": avg_review_length,
        "rating_distribution": ratings,
        "min_review_length": min(review_lengths) if review_lengths else 0,
        "max_review_length": max(review_lengths) if review_lengths else 0
    }
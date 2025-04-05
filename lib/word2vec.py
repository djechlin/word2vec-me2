"""
Library for Word2Vec functionality with different tokenization options
"""

import time
import logging
import multiprocessing
from typing import List, Optional, Union, Callable, Dict, Any, Tuple

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
import tiktoken

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Configure logging
logger = logging.getLogger(__name__)

def tokenize_with_nltk(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize."""
    return word_tokenize(text.lower())

def tokenize_with_tiktoken(text: str, model_name: str = "cl100k_base") -> List[str]:
    """Tokenize text using tiktoken."""
    encoding = tiktoken.get_encoding(model_name)
    token_integers = encoding.encode(text.lower())
    # Convert back to strings for word2vec
    return [str(token) for token in token_integers]

def process_corpus(
    corpus_path: str,
    tokenizer: Callable,
    max_articles: Optional[int] = None
) -> List[List[str]]:
    """Process a corpus file into tokenized sentences."""
    articles = []
    article_count = 0

    logger.info(f"Reading corpus from {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
        article = []
        for line in corpus_file:
            line = line.strip()

            # Check if line marks the end of an article
            if line == '':
                if article:
                    tokenized_article = tokenizer(' '.join(article))
                    if tokenized_article:
                        articles.append(tokenized_article)
                    article = []
                    article_count += 1

                    if max_articles is not None and article_count >= max_articles:
                        break

                    if article_count % 1000 == 0:
                        logger.info(f"Processed {article_count} articles")
            else:
                article.append(line)

        # Add the last article if there is one
        if article:
            tokenized_article = tokenizer(' '.join(article))
            if tokenized_article:
                articles.append(tokenized_article)

    logger.info(f"Processed a total of {len(articles)} articles")
    return articles

def train_word2vec(
    tokenized_corpus: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = None,
    sg: int = 1,  # 1 for skip-gram, 0 for CBOW
    epochs: int = 5
) -> Word2Vec:
    """Train a Word2Vec model on the tokenized corpus."""
    if workers is None:
        workers = multiprocessing.cpu_count()

    logger.info(f"Training Word2Vec model with: vector_size={vector_size}, window={window}, "
                f"min_count={min_count}, workers={workers}, sg={sg}, epochs={epochs}")

    start_time = time.time()
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")
    logger.info(f"Vocabulary size: {len(model.wv.key_to_index)}")

    return model

def evaluate_model(model: Word2Vec, word_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Perform basic evaluation on the trained model.

    Args:
        model: Trained Word2Vec model
        word_pairs: List of word pairs to evaluate similarity

    Returns:
        Dictionary of word pair similarities
    """
    results = {}

    for word1, word2 in word_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            pair_key = f"{word1}_{word2}"
            results[pair_key] = similarity
            logger.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        except KeyError as e:
            logger.warning(f"Cannot calculate similarity: {e}")

    # Find most similar words for a few examples
    example_words = ['computer', 'human', 'money', 'science']
    for word in example_words:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            logger.info(f"Words most similar to '{word}': {similar_words}")
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")

    return results

def get_model_statistics(model: Word2Vec) -> Dict[str, Any]:
    """
    Get statistics about a trained Word2Vec model

    Args:
        model: Trained Word2Vec model

    Returns:
        Dictionary with model statistics
    """
    return {
        "vocabulary_size": len(model.wv.key_to_index),
        "vector_size": model.vector_size,
        "window": model.window,
        "epochs": model.epochs,
        "algorithm": "skip-gram" if model.sg == 1 else "CBOW",
    }
"""
Library for Word2Vec functionality with different tokenization options
"""

import logging
import multiprocessing
import time
from collections.abc import Callable
from typing import Any

import nltk
import tiktoken
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download("punkt", quiet=True)

# Configure logging
logger = logging.getLogger(__name__)


def tokenize_with_nltk(text: str) -> list[str]:
    """
    Tokenizes text into words using NLTK's word_tokenize function.

    Processes the input text by converting to lowercase and splitting into
    linguistic tokens (words, punctuation) using natural language rules.
    This produces word-based tokens suitable for traditional word2vec training.

    Args:
        text: Raw text string to tokenize

    Returns:
        List of string tokens representing words and punctuation
    """
    return word_tokenize(text.lower())


def tokenize_with_tiktoken(text: str, model_name: str = "cl100k_base") -> list[str]:
    """
    Tokenizes text using OpenAI's tiktoken library and converts to string tokens.

    Processes text with tiktoken's BPE (byte pair encoding) tokenizer,
    converting to lowercase first. The integer token IDs are then converted
    to strings to make them compatible with word2vec's string-based vocabulary.
    This provides subword tokenization that can capture meaningful fragments.

    Args:
        text: Raw text string to tokenize
        model_name: Name of the tiktoken encoding model to use (default: cl100k_base)

    Returns:
        List of string tokens (converted from tiktoken's integer IDs)
    """
    encoding = tiktoken.get_encoding(model_name)
    token_integers = encoding.encode(text.lower())
    # Convert back to strings for word2vec
    return [str(token) for token in token_integers]


def process_corpus(
    corpus_path: str, tokenizer: Callable, max_articles: int | None = None
) -> list[list[str]]:
    """
    Processes a text corpus file into a list of tokenized documents.

    Reads a corpus file where documents are separated by blank lines,
    tokenizes each document using the provided tokenizer function,
    and returns a nested list structure suitable for Word2Vec training.
    Respects document boundaries to ensure that training contexts don't
    cross between unrelated texts.

    Args:
        corpus_path: Path to the corpus text file
        tokenizer: Function that converts a text string into a list of tokens
        max_articles: Optional limit on the number of documents to process

    Returns:
        List of tokenized documents, where each document is a list of tokens
    """
    articles = []
    article_count = 0

    logger.info(f"Reading corpus from {corpus_path}")
    with open(corpus_path, encoding="utf-8") as corpus_file:
        article = []
        for line in corpus_file:
            line = line.strip()

            # Check if line marks the end of an article
            if line == "":
                if article:
                    tokenized_article = tokenizer(" ".join(article))
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
            tokenized_article = tokenizer(" ".join(article))
            if tokenized_article:
                articles.append(tokenized_article)

    logger.info(f"Processed a total of {len(articles)} articles")
    return articles


def train_word2vec(
    tokenized_corpus: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int | None = None,
    sg: int = 1,  # 1 for skip-gram, 0 for CBOW
    epochs: int = 5,
) -> Word2Vec:
    """
    Trains a Word2Vec model on a tokenized corpus of documents.

    Creates and trains a gensim Word2Vec model using the provided parameters.
    Automatically determines the optimal number of worker threads based on
    available CPU cores if not specified. Logs training progress and model
    statistics. The resulting model captures semantic relationships between
    tokens in the training corpus.

    Args:
        tokenized_corpus: List of documents, where each document is a list of tokens
        vector_size: Dimensionality of the word vectors (default: 100)
        window: Maximum distance between target and context words (default: 5)
        min_count: Ignores tokens with total frequency lower than this (default: 5)
        workers: Number of CPU threads to use (default: all available cores)
        sg: Training algorithm: 1 for skip-gram, 0 for CBOW (default: 1)
        epochs: Number of training passes over the corpus (default: 5)

    Returns:
        Trained Word2Vec model with word vectors and vocabulary
    """
    if workers is None:
        workers = multiprocessing.cpu_count()

    logger.info(
        f"Training Word2Vec model with: vector_size={vector_size}, window={window}, "
        f"min_count={min_count}, workers={workers}, sg={sg}, epochs={epochs}"
    )

    start_time = time.time()
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")
    logger.info(f"Vocabulary size: {len(model.wv.key_to_index)}")

    return model


def evaluate_model(model: Word2Vec, word_pairs: list[tuple[str, str]]) -> dict[str, float]:
    """
    Evaluates a trained Word2Vec model by analyzing semantic similarities.

    Performs two types of evaluation on the model:
    1. Calculates cosine similarity between specific word pairs to test
       semantic relationships (like analogies and associations)
    2. Finds most similar words to example keywords to demonstrate
       the model's learned semantic space

    Logs results for inspection and returns similarity scores for further analysis.

    Args:
        model: Trained Word2Vec model with loaded word vectors
        word_pairs: List of word pairs (tuples) to measure similarity between

    Returns:
        Dictionary mapping word pair keys (format: "word1_word2") to similarity scores
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
    example_words = ["computer", "human", "money", "science"]
    for word in example_words:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            logger.info(f"Words most similar to '{word}': {similar_words}")
        except KeyError:
            logger.warning(f"Word '{word}' not in vocabulary")

    return results


def get_model_statistics(model: Word2Vec) -> dict[str, Any]:
    """
    Extracts key configuration and size statistics from a trained Word2Vec model.

    Generates a summary of the model's characteristics including vocabulary size,
    vector dimensions, training parameters, and algorithm choice. This information
    is useful for model documentation, comparison between different training runs,
    and diagnosing training issues.

    Args:
        model: Trained Word2Vec model to analyze

    Returns:
        Dictionary containing model statistics with the following keys:
        - vocabulary_size: Number of unique tokens in the model's vocabulary
        - vector_size: Dimensionality of the word vectors
        - window: Context window size used during training
        - epochs: Number of training iterations performed
        - algorithm: Either "skip-gram" or "CBOW" depending on training method
    """
    return {
        "vocabulary_size": len(model.wv.key_to_index),
        "vector_size": model.vector_size,
        "window": model.window,
        "epochs": model.epochs,
        "algorithm": "skip-gram" if model.sg == 1 else "CBOW",
    }

#!/usr/bin/env python3
"""
Word2Vec training script with multiple tokenization options
"""

import argparse
import os
import time
import logging
from typing import List, Optional, Union, Callable
import multiprocessing

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
import tiktoken

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

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

    logging.info(f"Reading corpus from {corpus_path}")
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
                        logging.info(f"Processed {article_count} articles")
            else:
                article.append(line)

        # Add the last article if there is one
        if article:
            tokenized_article = tokenizer(' '.join(article))
            if tokenized_article:
                articles.append(tokenized_article)

    logging.info(f"Processed a total of {len(articles)} articles")
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

    logging.info(f"Training Word2Vec model with: vector_size={vector_size}, window={window}, "
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
    logging.info(f"Training completed in {elapsed:.2f} seconds")
    logging.info(f"Vocabulary size: {len(model.wv.key_to_index)}")

    return model

def evaluate_model(model: Word2Vec, word_pairs: List[tuple]) -> None:
    """Perform basic evaluation on the trained model."""
    for word1, word2 in word_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            logging.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        except KeyError as e:
            logging.warning(f"Cannot calculate similarity: {e}")

    # Find most similar words for a few examples
    example_words = ['computer', 'human', 'money', 'science']
    for word in example_words:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            logging.info(f"Words most similar to '{word}': {similar_words}")
        except KeyError:
            logging.warning(f"Word '{word}' not in vocabulary")

def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec with different tokenization methods")
    parser.add_argument('--corpus', type=str, required=True, help='Path to corpus file')
    parser.add_argument('--tokenizer', type=str, choices=['nltk', 'tiktoken'], default='nltk',
                        help='Tokenizer to use: nltk or tiktoken')
    parser.add_argument('--output', type=str, default='word2vec_model',
                        help='Path to save the trained model')
    parser.add_argument('--max_articles', type=int, default=None,
                        help='Maximum number of articles to process')
    parser.add_argument('--vector_size', type=int, default=100,
                        help='Dimensionality of the word vectors')
    parser.add_argument('--window', type=int, default=5,
                        help='Maximum distance between the current and predicted word')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum count of words to consider for training')
    parser.add_argument('--sg', type=int, default=1, choices=[0, 1],
                        help='Training algorithm: 1 for skip-gram, 0 for CBOW')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')

    args = parser.parse_args()

    # Select tokenizer
    tokenizer = tokenize_with_nltk if args.tokenizer == 'nltk' else tokenize_with_tiktoken
    tokenizer_name = args.tokenizer

    # Process corpus
    tokenized_corpus = process_corpus(
        corpus_path=args.corpus,
        tokenizer=tokenizer,
        max_articles=args.max_articles
    )

    if not tokenized_corpus:
        logging.error("No valid articles found in corpus. Exiting.")
        return

    # Train model
    model = train_word2vec(
        tokenized_corpus=tokenized_corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        epochs=args.epochs
    )

    # Basic evaluation
    word_pairs = [
        ('man', 'woman'),
        ('king', 'queen'),
        ('computer', 'laptop'),
        ('good', 'bad'),
        ('cat', 'dog')
    ]
    evaluate_model(model, word_pairs)

    # Save model
    output_path = f"{args.output}_{tokenizer_name}.model"
    model.save(output_path)
    logging.info(f"Model saved to {output_path}")

    # Save word vectors for easier use
    vector_path = f"{args.output}_{tokenizer_name}.wordvectors"
    model.wv.save(vector_path)
    logging.info(f"Word vectors saved to {vector_path}")

if __name__ == "__main__":
    main()
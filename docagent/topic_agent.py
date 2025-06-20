from docagent import topic_assignment as ta, topic_discovery as td
import asyncio
import pandas as pd
import time

async def full_topic_analysis(corpus: pd.DataFrame, corpus_name: str, min_topics: int = 7, max_topics: int = 15,
                              model_temperature: float = 0.7, chunk_size = 20) -> tuple:
    """
    Perform full topic analysis including discovery and assignment of topics to documents.  Uses models as specified
    in the environment variables for topic discovery and assignment.
    
    Args:
        corpus (pd.DataFrame): DataFrame containing documents with 'docID' and 'content' columns.
        corpus_name (str): Name of the corpus for processing.
        min_topics (int): Minimum number of topics to discover.
        max_topics (int): Maximum number of topics to discover.
        model_temperature (float): Temperature for the topic discovery and assignment models.
        chunk_size (int): Size of document chunks for processing assignments.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame of discovered topics.
            - pd.DataFrame: DataFrame of documents with assigned primary, secondary and tertiary topics.
    """
    
    start = time.time()

    # Discover topics
    discovered_topics = await td.discover_topics(
        model_temperature = model_temperature,
        corpus = corpus, 
        min_topics = min_topics, 
        max_topics = max_topics)
    
    # Assign topics to documents
    doc_assignment = await ta.assign_topics(
        model_temperature = model_temperature,
        corpus_name = corpus_name,
        corpus = corpus,
        topics = discovered_topics,
        chunk_size = chunk_size
    )

    timer = time.time() - start

    print(f"Agent has completed topic discovery and assignment. Total time taken: {int(timer)} seconds.")

    return discovered_topics, doc_assignment
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import time
import asyncio
import os
import json
import pandas as pd

load_dotenv()

analytic_model = os.getenv("DEFAULT_ANALYTIC_MODEL", "claude-2")

class TopicDiscovery(BaseModel):
    topicID: int = Field(..., description="Unique identifier for the topic")
    topicName: str = Field(..., description="Name of the topic")
    topicDescription: str = Field(..., description="Description of the topic")
    topicKeywords: list[str] = Field(..., description="List of keywords associated with the topic")
    topicExamples: list[int] = Field(..., description="List of the docIDs of 1-3 example documents related to the topic")
    topicPrevalence: float = Field(
        ...,
        description="Prevalence of the topic in the corpus, as a percentage of documents that are related to the topic",
    )

async def discover_topics(corpus: pd.DataFrame, model: str = analytic_model, min_topics: int = 7, max_topics: int =15) -> pd.DataFrame:
    """
    Discover topics from a DataFrame of documents using the specified model.
    
    Args:
        docs (pd.DataFrame): DataFrame containing documents with 'docID' a 'content' column.
        model (str): The model to use for topic discovery.
        min_topics (int): Minimum number of topics to discover.
        max_topics (int): Maximum number of topics to discover.
    
    Returns:
        pd.DataFrame: DataFrame containing discovered topics.
    """

    start = time.time()
    client = Anthropic()
    
    concat = str(corpus['docID']) + " " + corpus['content']
    doc_string = '\n '.join(concat.tolist())

    prompt = f"""
    You are an expert in topic discovery. Analyze the following numbered documents and identify distinct topics.
    Return a minimum of {min_topics} and a maximum of {max_topics} distinct topics.
    Return a Python list of JSONs, each containing the following keys:
    - `topicID`: Value is unique identifier for the topic, increasing integers starting from 1
    - `topicName`: Value is an appropriate name for the topic
    - `topicDescription`: Value is a description of the topic in 1-2 sentences
    - `topicKeywords`: Value is a list of keywords associated with the topic
    - `topicExamples`: Value is a list of example docIDs related to the topic from the list of documents provided
    - `topicPrevalence`: Value is a float between 0 and 1, representing the prevalence of the topic in the corpus, as a percentage of all documents provided to you.

    The list of documents follows now:
    
    {doc_string}
    """

    print(f"Starting topic discovery from {len(corpus)} documents using model {model}...")
    response = client.messages.create(
        model=model,
        max_tokens=8000,    
        system="You are an expert in determining distinct topics from a set of documents.", 
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": '[{"topicID": 1, "topicName": "'}
        ]
    )
    
    raw_response = '[{"topicID": 1, "topicName": "' + response.content[0].text
    docs = json.loads(raw_response)
    new_docs = []
    for i, doc in enumerate(docs):
        new_docs.append(TopicDiscovery(**doc))
    timer = time.time() - start
    print(f"Topic discovery complete. Discovered {len(new_docs)} topics in {int(timer)} seconds.")
    return pd.DataFrame([res.model_dump() for res in new_docs]).sort_values(by='topicPrevalence', ascending=False).reset_index(drop=True)
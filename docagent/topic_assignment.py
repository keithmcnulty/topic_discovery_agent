from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import time
import asyncio
import os
import json
import pandas as pd

load_dotenv()
assignment_model = os.getenv("DEFAULT_ASSIGNMENT_MODEL", "claude-2")

class TopicAssignment(BaseModel):
    docID: int = Field(..., description="Unique identifier for the document")
    docContent: str = Field(..., description="Content of the document")
    primaryTopicID: int = Field(..., description="TopicID for the primary assigned topic")
    primaryTopicName: str = Field(..., description="Name of the primary assigned topic")
    secondaryTopicID: int = Field(..., description="TopicID for the secondary assigned topic")
    secondaryTopicName: str = Field(..., description="Name of the secondary assigned topic")
    tertiaryTopicID: int = Field(..., description="TopicID for the tertiary assigned topic")
    tertiaryTopicName: str = Field(..., description="Name of the tertiary assigned topic")

def get_docs_from_corpus(docIDs: list[int], corpus: pd.DataFrame) -> str:
    """
    Get a specified list of documents from corpus DataFrame and convert into a formatted string for processing.
    
    Args:
        corpus (pd.DataFrame): DataFrame containing documents with 'docID' and 'content' columns.
        docIDs (list[int]): List of document IDs to retrieve from the corpus.
        
    Returns:
        str: Formatted string of documents.
    """

    concat = []
    for i in docIDs:
        string = str(corpus['docID'][corpus['docID'] == i].item()) + '. ' + str(corpus['content'][corpus['docID'] == i].item())
        concat.append(string)
    return '\n '.join(concat)

tool_payload = [
    {
        "name": "get_docs_from_corpus",
        "description": "Get a specified list of documents from corpus DataFrame and convert into a formatted string for processing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "docIDs": {
                    "type": "string",
                    "description": "The list of the document IDs to retrieve from the corpus",
                },
                "corpus": {
                    "type": "string",
                    "description": "The name of the corpus DataFrame containing documents with 'docID' and 'content' columns",
                }
            },
            "required": ["docIDs", "corpus"]
        }
    }
]

# Map function names to actual functions
tool_map = {
    "get_docs_from_corpus": get_docs_from_corpus
}

async def _batch_assign_topics(docIDs: list[int], corpus_name: str, corpus: pd.DataFrame, topics: pd.DataFrame,
                                model:str = assignment_model) -> pd.DataFrame:
    """
    Process a list of document IDs and assign each to one or more topics within the original corpus dataframe.
    
    Args:
        docIDs (list[int]): List of document IDs to process.
        corpus (pd.DataFrame): DataFrame containing documents with 'docID' and 'content' columns.
    
    Returns:
        pd.DataFrame: DataFrame containing assigned topics for each document.
    """

    corpus_dict = {corpus_name: corpus}
    
    concat = []
    for i in range(len(topics)):
        string = "topicID: " + str(topics['topicID'][i]) + "; topicName: " + topics['topicName'][i] + "; topicDescription: " + topics['topicDescription'][i] + "; topicKeywords: " + str(topics['topicKeywords'][i]) + "; topicExamples: " + str(topics['topicExamples'][i])
        concat.append(string)

    topic_string = "\n ".join(concat)

    document_string = ', '.join([str(element) for element in docIDs])
    
    client = Anthropic()

    prompt = f"""
    You are an expert in topic assignment and you are required to assign documents to a set of topics based on their content.
    There are {len(topics)} topics and they are defined as follows:
    {topic_string}  

    You have a tool available to allow you to obtain the documents that you need to analyze from a DataFrame.  
    This tool will provide you with a list of the documents, each one labelled by an integer document ID.

    Once you have retrieved the documents, analyze them and assign each document to up to three topics based
    on the content of the document.
    If the document is empty, too short or otherwise not related to any of the topics, assign a topicID of 0.
    Return a Python list of JSONs, each containing the following keys:
    - `docID`: Value is the document ID for the document as it was provided to you
    - `docContent`: Value is the content of the document as it was provided to you
    - `primaryTopicName`: Value is the name of the most closely related topic exactly as it appears in the `topicName` field provided earlier, or "No topic assigned" if no topic is assigned
    - `primaryTopicID`: Value is the `topicID` for the most closely related topic, ensuring the `topicID` matches the `topicName` as provided earlier.  Record the `primaryTopicID` as 0 if no topic is assigned
    - `secondaryTopicName`: Value is the name of the second most closely related topic exactly as it appears in the `topicName` field provided earlier, or "No topic assigned" if no secondary topic is assigned
    - `secondaryTopicID`: Value is the `topicID` for the second most closely related topic, ensuring the `topicID` matches the `topicName` as provided earlier.  Record the `secondaryTopicID` as 0 if no secondary topic is assigned
    - `tertiaryTopicName`: Value is the name of the third most closely related topic exactly as it appears in the `topicName` field provided earlier, or "No topic assigned" if no tertiary topic is assigned
    - `tertiaryTopicID`: Value is the `topicID` for the third most closely related topic, ensuring the `topicID` matches the `topicName` as provided earlier.  Record the `tertiaryTopicID` as 0 if no tertiary topic is assigned

    Here is the list of document IDs that you need to assign topics to: {document_string}.  
    These documents can be found in the corpus named {corpus_name}.    
    """
    
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system="You are an expert in topic assignment and you are required to assign documents to a set of topics based on their content",
        messages=[
            {"role": "user", "content": prompt},
        ],
        tools=tool_payload,
    )

    print(f"Document assignment under way using model {model}...")

    while response.stop_reason == "tool_use":
        print("Tool use detected.  Obtaining the documents from the corpus...")
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool = tool_map[tool_name]
        tool_input_new = tool_input.copy()
        tool_input_new['corpus'] = corpus_dict[tool_input_new['corpus']]
        tool_input_new['docIDs'] = json.loads('[' + tool_input_new['docIDs'] + ']')
        tool_result = tool(**tool_input_new) + "\n" + """
        Return only a list of JSONs, each in the following example format with no other text:
        {"docID": 1, "docContent": "Some text", "primaryTopicID": 2. "primaryTopicName": "Some topic name",
        "secondaryTopicID": 3, "secondaryTopicName": "Another topic name", "tertiaryTopicID": 4, "tertiaryTopicName": "Yet another topic name"}
        """
        
        print("Documents obtained successfully, continuing with topic assignment...")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result,
                    }
                ],
            },
            {"role": "assistant", "content": '[{"docID":'}
        ]

        response = client.messages.create(
            model = model,
            max_tokens=8000,
            messages=messages
        )

        raw_response ='[{"docID": ' + response.content[0].text
        docs = json.loads(raw_response)
        new_docs = []
        for i, doc in enumerate(docs):
            new_docs.append(TopicAssignment(**doc))
        print(f"Processed {len(new_docs)} documents with tool use.")
        return pd.DataFrame([res.model_dump() for res in new_docs])
    else:
        print("An error occurred while processing the document assignment.")
    

async def assign_topics(corpus_name: str, corpus: pd.DataFrame, topics: pd.DataFrame,
                        model:str = assignment_model, chunk_size: int = 20) -> pd.DataFrame:
    
    """
    Assign topics to documents in the corpus based on the provided topics DataFrame.

    Args:
        corpus_name (str): Name of the corpus for processing.
        corpus (pd.DataFrame): DataFrame containing documents with 'docID' and 'content' columns.
        topics (pd.DataFrame): DataFrame containing topics with 'topicID', 'topicName', etc.
        model (str): The model to use for topic assignment.
        chunk_size (int): Size of document chunks for processing assignments.
    Returns:
        pd.DataFrame: DataFrame containing assigned topics for each document.
    """

    start = time.time()
    print(f"Commencing assignment of {len(corpus)} documents to {len(topics)} topics. This will be done in chunks of {chunk_size} documents at a time...")

    # Assign topics to documents
    docs_total = range(1, len(corpus) + 1)
    doc_chunks = [docs_total[x:x + chunk_size] for x in range(0, len(docs_total), chunk_size)]
    lst = []
    for chunk in doc_chunks:
        print(f"Assigning topics for docs {chunk[0]} to {chunk[-1]}...")
        assigned_docs = await _batch_assign_topics(
            docIDs = chunk,
            corpus_name = corpus_name,
            corpus = corpus,
            topics = topics
        )
        lst.append(assigned_docs) 
    timer = time.time() - start
    print(f"Topic assignment complete for {len(corpus)} documents in {int(timer)} seconds.")
    return pd.concat(lst, ignore_index=True)
   
    
   
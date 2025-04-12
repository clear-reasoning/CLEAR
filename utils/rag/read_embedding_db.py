import pickle
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Callable, Any, Dict
from utils.rag.embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel, EnvironmentEmbeddingModel
from utils.rag.similarity_functions import cosine_similarity, l2_loss, compare_dataframes
from utils.df_utils import get_formatted_df_for_llm

class RAG_Database:
    """A class for reading and managing a RAG (Retrieval-Augmented Generation) database from a pickle file.

    The pickle file contains a dictionary with the following structure:
    
    Attributes:
        embedding_index (List[np.ndarray]): Stores the embeddings of the database. When performing similarity 
            search, the query will be embedded and compared against these embeddings. Dimensions are 
            (N_ROWS_IN_DATABASE, EMBEDDING_DIMENSION). This is a numpy array.
            
            Example:
                >>> self.db_data["embedding_index"][0]
                array([-0.001,  0.002, -0.003, ...,  0.004, -0.005,  0.006])
                >>> self.db_data["embedding_index"].shape
                (1000, 1536)  # number of rows in the database x 1536 (embedding dimension)

        json_contents (List[str]): Stores the stringified JSON documents of the database. After computing 
            similarity scores with the embedding index, documents are retrieved by indexing into this attribute.
            
            Example:
                >>> self.db_data["json_contents"][0]
                {
                    'environment_key': '(headway=0.5, speed=0.4, leader_speed=0.3)', 
                    'environment_explanation': 'We see that the leader speed is slowing down...'
                }

        embedding_key (str): The key of the embedding model used to create the database.
            Example: "environment_embedding"

        file_path (str): The path to the file used to create the database.
            Example: "environment_db_examples.json"

        db_rows (List[dict]): The rows of the database. Note that this attribute is optional and may not 
            always be present in the pickle file.
    """
    
    def __init__(self, db_pkl_path: str):
        self.db_pkl_path = db_pkl_path
        self.db_data = None
        
        with open(db_pkl_path, 'rb') as f:
            self.db_data = pickle.load(f)
                
        # Retrieving the embedding index and json contents
        self.embedding_index: np.ndarray = self.db_data['embedding_index']
        self.json_contents: List[Dict[str, Any]] = self.db_data['json_contents']
        self.embedding_key: str = self.db_data['embedding_key']
        self.file_path: str = self.db_data['file_path']
        
        # Note: db_rows is not always present in the pickle file.
        self.db_rows = self.db_data.get('db_rows', [])
    
    def replace_document(self, document_index: int, new_json_content: Dict[str, Any], embedding_model: BaseEmbeddingModel) -> None:
        """Replace a document in the database
        
        Args:
            document_index (int): The index of the document to replace
            new_json_content (Dict[str, Any]): The new content for the document
            embedding_model (BaseEmbeddingModel): The embedding model to use
        """
        assert document_index < len(self.embedding_index), "Document index is out of range"
        assert isinstance(new_json_content, dict), "Document contents must be a dictionary"
        
        self.embedding_index[document_index] = embedding_model.get_embedding(new_json_content['environment_key'])
        self.json_contents[document_index] = new_json_content
        
    def add_document(self, new_json_content: Dict[str, Any], embedding_model: BaseEmbeddingModel) -> None:
        """Add a document to the database
        
        Args:
            new_json_content (Dict[str, Any]): The content for the new document
            embedding_model (BaseEmbeddingModel): The embedding model to use
        """
        new_embedding = embedding_model.get_embedding(new_json_content['environment_key'])
        self.embedding_index = np.vstack([self.embedding_index, new_embedding])
        self.json_contents.append(new_json_content)
    
    def remove_document(self, document_index: int) -> None:
        """Remove a document from the database
        
        Args:
            document_index (int): The index of the document to remove
        """
        self.embedding_index = np.delete(self.embedding_index, document_index, axis=0)
        self.json_contents.pop(document_index)
    
    def save_database(self, save_path: str = None) -> None:
        """Save the database to a pickle file
        
        Args:
            save_path (str, optional): The path to save the database. If None, uses the original db_pkl_path
        """
        if save_path is None:
            save_path = self.db_pkl_path
        
        # Update db_data with current state before saving
        self.db_data = {
            'embedding_index': self.embedding_index,
            'json_contents': self.json_contents,
            'embedding_key': self.embedding_key,
            'file_path': self.file_path        
        }
        
        # Saving db_rows if it exists
        if 'db_rows' in self.db_data:
            self.db_data['db_rows'] = self.db_rows
                
        with open(save_path, 'wb') as f:
            pickle.dump(self.db_data, f)
        
        print(f"Database saved successfully to {save_path}")
    

    def get_top_k_documents(self, query: str, embedding_model: BaseEmbeddingModel, k: int, similarity_function: Callable[np.ndarray, Any], apply_normalization: bool = False) -> List[Tuple[float, np.ndarray, str]]:
        """Get the top k documents based on a similarity function
        
        Args:
            k (int): The number of top documents to retrieve
            similarity_function (Callable): A function that takes an embedding and a document and returns a similarity score. The higher the score, the more similar the document is to the embedding.
        
        Returns:
            List[Tuple[np.ndarray, str]]: A list of tuples containing (similarity score, embedding, document) sorted by similarity. The highest similarity score is first (element 0 in the list).
        """
        # Convert the query to an embedding
        query_embedding = embedding_model.get_embedding(query)
        
        # Calculate similarity scores. 
        similarity_scores = [
            similarity_function(query_embedding, doc_embedding) for doc_embedding in self.embedding_index
        ]
        
        # Normalize the similarity scores between 0 and 1 if apply_normalization is True
        if apply_normalization:
            max_score = max(similarity_scores)
            min_score = min(similarity_scores)
            range_score = max_score - min_score
            
            # only apply if the range is not zero, avoiding division by zero.
            if range_score > 0:
                similarity_scores = [(score - min_score) / range_score for score in similarity_scores]
            else:
                similarity_scores = [0.0] * len(similarity_scores)
        
        scored_docs = [
            (score, doc_embedding, doc, index)
            for score, doc_embedding, doc, index in zip(similarity_scores, self.embedding_index, self.json_contents, range(len(self.embedding_index)))
        ]
        
        # Sort documents by similarity score in descending order
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        return sorted_docs[:k]
    
    def get_top_k_situations(self, first_steps: pd.DataFrame, embedding_model: BaseEmbeddingModel, k: int, columns=None, apply_normalization: bool = False) -> List[Tuple[float, np.ndarray, str]]:
        """Get the top k situations based on a similarity function on the provided timesteps
        
        Args:
            k (int): The number of top situations to retrieve
            similarity_function (Callable): A function that takes an embedding and a document and returns a similarity score. The higher the score, the more similar the document is to the embedding.
        
        Returns:
            List[Tuple[np.ndarray, str]]: A list of tuples containing (similarity score, embedding, situation) sorted by similarity. The highest similarity score is first (element 0 in the list). Note: the embedding is the embedding of the dataframe of the steps and the explanation, the situtation is the situation as a text (steps and explanation).
        """
        # Calculate similarity scores. 
        similarity_scores = [
            compare_dataframes(first_steps, situation['last_df'], columns) for situation in self.db_rows
        ]

        # Normalize the similarity scores between 0 and 1 if apply_normalization is True
        if apply_normalization:
            max_score = max(similarity_scores)
            min_score = min(similarity_scores)
            range_score = max_score - min_score
            
            # only apply if the range is not zero, avoiding division by zero.
            if range_score > 0:
                similarity_scores = [(score - min_score) / range_score for score in similarity_scores]
            else:
                similarity_scores = [0.0] * len(similarity_scores)

        scored_docs = [
            (score, situation_embedding, row)
            for score, situation_embedding, row in zip(similarity_scores, self.embedding_index, self.json_contents)
        ]
        
        # Sort documents by similarity score in descending order
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

        # get for the top k situations: (score, full embedding, full situation as a text)
        for i in range(k):
            # dataframe to text and then to embedding
            text_df_last = get_formatted_df_for_llm(pd.DataFrame(sorted_docs[i][2]["last_timesteps"]), precision=2)
            text_df_next = get_formatted_df_for_llm(pd.DataFrame(sorted_docs[i][2]["next_timesteps"]), precision=2)
            text_situation = "DataFrame of the last " + str(len(sorted_docs[i][2]["last_timesteps"])) + "timesteps" + "\n" + text_df_last + "\n" + sorted_docs[i][2]["explanation"] + "\n" + "DataFrame of the next " + str(len(sorted_docs[i][2]["last_timesteps"])) + " timesteps" + "\n" + text_df_next

            sorted_docs[i] = (sorted_docs[i][0], sorted_docs[i][1], text_situation)
        
        return sorted_docs[:k]
    
    def save_as_json(self, save_path: str = None):
        """Save the database as a JSON file
        
        Args:
            save_path (str, optional): The path to save the JSON file. If None, uses the original db_pkl_path
        """
        if save_path is None:
            save_path = self.db_pkl_path.replace('.pkl', '.json')
            
        # only saving self.json_contents as the JSON file
        with open(save_path, 'w') as f:
            json.dump(self.json_contents, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    # Example usage of using the RAG_Database
    db = RAG_Database("environment_db_examples.pkl")
    embedding_model = EnvironmentEmbeddingModel()
    
    # Adding a new document to the database
    db.add_document({
            "environment_key": "(headway=0.3, speed=0.12, leader_speed=0.7)",
            "environment_explanation": "This is a new document"
            }, embedding_model)
    
    # top_k_docs = db.get_top_k_documents(query="What are popular techniques in computer vision?", k=3, embedding_model=OpenAIEmbeddingModel(), similarity_function=cosine_similarity)
    top_k_docs = db.get_top_k_documents(query="(headway=0.3, speed=0.12, leader_speed=0.7)", k=3, embedding_model=EnvironmentEmbeddingModel(), similarity_function=l2_loss)

    for doc in top_k_docs:
        print(doc)
        
    # replace the first document with a new one
    db.replace_document(0, {
        "environment_key": "(headway=0.3, speed=0.12, leader_speed=0.7)",
        "environment_explanation": "This is a new document"
    }, embedding_model)
    
    print("Replaced the first document")
    
    top_k_docs = db.get_top_k_documents(query="(headway=0.3, speed=0.12, leader_speed=0.7)", k=3, embedding_model=EnvironmentEmbeddingModel(), similarity_function=l2_loss)
    for doc in top_k_docs:
        print(doc)
        
    # adding more documents with the same key before saving the database 
    db.add_document({
        "environment_key": "(headway=0.3, speed=0.12, leader_speed=0.7)",
        "environment_explanation": "This is a new document"
    }, embedding_model)
    
    db.add_document({
        "environment_key": "(headway=0.3, speed=0.12, leader_speed=0.7)",
        "environment_explanation": "This is a new document"
    }, embedding_model)
    
    db.save_database("environment_db_examples_new.pkl")
    db = RAG_Database("environment_db_examples_new.pkl")
    
    # remove the first document
    print("Removed the first document")
    
    top_k_docs = db.get_top_k_documents(query="(headway=0.3, speed=0.12, leader_speed=0.7)", k=3, embedding_model=EnvironmentEmbeddingModel(), similarity_function=l2_loss)
    for doc in top_k_docs:
        print(doc)
    
    # save the database as a JSON file
    db.save_as_json("environment_db_examples_new.json")



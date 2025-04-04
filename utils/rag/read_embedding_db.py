import pickle
import numpy as np
from typing import List, Tuple, Callable, Any
from utils.rag.embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel, EnvironmentEmbeddingModel
from utils.rag.similarity_functions import cosine_similarity, l2_loss

class RAG_Database:
    def __init__(self, db_pkl_path: str):
        self.db_pkl_path = db_pkl_path
        self.db_data = None
        
        with open(db_pkl_path, 'rb') as f:
            self.db_data = pickle.load(f)
        
        # retrieving the embedding index and json contents
        self.embedding_index = self.db_data['embedding_index']
        self.json_contents = self.db_data['json_contents']
        self.embedding_key = self.db_data['embedding_key']
        self.file_path = self.db_data['file_path']
    
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
            (score, doc_embedding, doc)
            for score, doc_embedding, doc in zip(similarity_scores, self.embedding_index, self.json_contents)
        ]
        
        # Sort documents by similarity score in descending order
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        return sorted_docs[:k]
    
if __name__ == "__main__":
    # Example usage of using the RAG_Database
    db = RAG_Database("environment_db_examples.pkl")
    # top_k_docs = db.get_top_k_documents(query="What are popular techniques in computer vision?", k=3, embedding_model=OpenAIEmbeddingModel(), similarity_function=cosine_similarity)
    top_k_docs = db.get_top_k_documents(query="(headway=0.3, speed=0.12, leader_speed=0.7)", k=3, embedding_model=EnvironmentEmbeddingModel(), similarity_function=l2_loss)

    for doc in top_k_docs:
        print(doc)

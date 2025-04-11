import json
import numpy as np
import asyncio
import argparse
import pandas as pd
import pickle
import os
from typing import List, Tuple, Dict, Any
from utils.rag.embedding_models import OpenAIEmbeddingModel, EnvironmentEmbeddingModel
from utils.df_utils import get_formatted_df_for_llm

class CreateEmbeddingDB:
    """Class to create and manage an embedding database from a JSON file. Saves the embeddings to a .pkl file for later use."""
    @classmethod
    async def create(cls, file_path: str, embedding_key: str, embedding_model, pkl_path: str = None, type_data_json: bool = False):
        """Async factory method to create and initialize the EmbeddingDB"""
        instance = cls(file_path, embedding_key, embedding_model, pkl_path, type_data_json)
        if type_data_json:
            await instance._initialize_2()
        else:
            await instance._initialize()
        return instance
        
    def __init__(self, file_path: str, embedding_key: str, embedding_model, pkl_path: str = None, type_data_json: bool = False):
        # this is a JSON file path that we will read to create our "embedding_index"
        self.file_path = file_path
        self.embedding_index: np.ndarray = None
        self.db_rows = None
        self.pkl_path = pkl_path
        self.type_data_json = type_data_json
        
        with open(file_path, 'r') as f:
            self.json_contents = json.load(f)
        
        # Saving the embedding key and the embedding model
        self.embedding_key = embedding_key
        self.embedding_model = embedding_model
    
    async def _initialize(self):
        """Async initialization method to create embeddings"""
        # Iterate through the JSON file contents, creating the index of embeddings
        contents_to_vectorize = [row[self.embedding_key] for row in self.json_contents]
        
        # Asyncly get the embeddings for the contents
        embeddings = await self.embedding_model.get_embeddings_batch_async(contents_to_vectorize)
        
        # Store the embeddings
        self.embedding_index = np.array([emb for emb, _ in embeddings])
        
        # Save to pickle file
        self.save_to_pickle()
        print(f"Embeddings saved to {self.pkl_path}")
    
    async def _initialize_2(self):
        """Async initialization to store the timesteps in a dataframe and the token embeddings of the explanation text."""
        # Embed only environment_explanation
        contents_to_vectorize = []
        for row in self.json_contents:
            text_df_last = get_formatted_df_for_llm(pd.DataFrame(row["last_timesteps"]), precision=2)
            text_df_next = get_formatted_df_for_llm(pd.DataFrame(row["next_timesteps"]), precision=2)
            text_situation = "DataFrame of the last " + str(len(row["last_timesteps"])) + "timesteps" + "\n" + text_df_last + "\n" + row["explanation"] + "\n" + "DataFrame of the next " + str(len(row["last_timesteps"])) + " timesteps" + "\n" + text_df_next
            contents_to_vectorize.append(text_situation)

        # Asyncly get the embeddings for the contents
        embeddings = await self.embedding_model.get_embeddings_batch_async(contents_to_vectorize)

        # Store the embeddings
        self.embedding_index = np.array([emb for emb, _ in embeddings])

        data_to_store = []
        for i, row in enumerate(self.json_contents):
            # Create a dataframe for the environment key
            last_df = pd.DataFrame(row["last_timesteps"])
            next_df = pd.DataFrame(row["next_timesteps"])
            
            # Add the embedding to the data to store
            data_to_store.append({
                'last_df': last_df,
                'next_df': next_df,
                'embedding': self.embedding_index[i]
            })

        # Store the data
        self.db_rows = data_to_store

        # Save to pickle file
        self.save_to_pickle()
        print(f"Embeddings saved to {self.pkl_path}")
    
    def save_to_pickle(self):
        """Save the necessary data to a pickle file (excluding the model)"""
        data_to_save = {
            'embedding_index': self.embedding_index,
            'json_contents': self.json_contents,
            'embedding_key': self.embedding_key,
            'file_path': self.file_path,
            'db_rows': self.db_rows,
        }
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    @classmethod
    def load_from_pickle(cls, pkl_path: str, embedding_model):
        """Load data from pickle and create a new EmbeddingDB instance"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            data['file_path'], 
            data['embedding_key'], 
            embedding_model,
            pkl_path
        )
        
        # Set the loaded data
        instance.embedding_index = data['embedding_index']
        instance.json_contents = data['json_contents']
        
        return instance
    
    def get_top_k_embeddings(self, query: str, k: int = 5) -> List[Tuple[np.ndarray, str]]:
        """Get the top K embeddings for a given query"""
        # Get the embedding for the query
        query_embedding = self.embedding_model.get_embedding(query)
        
        # Calculate cosine similarity
        similarities = np.dot(self.embedding_index, query_embedding) / (np.linalg.norm(self.embedding_index, axis=1) * np.linalg.norm(query_embedding))
        
        # Get the top K indices
        top_k_indices = np.argsort(similarities)[-k:]
        
        # Return the top K embeddings and their corresponding texts
        return [(self.embedding_index[i], self.json_contents[i][self.embedding_key]) for i in top_k_indices]

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create embeddings database from JSON data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSON data file')
    parser.add_argument('--embedding_key', type=str, default='text_content', 
                       help='Field name within the JSON file that we will serve the embedding key')
    parser.add_argument('--pkl_name', type=str, default=None, 
                       help='Name of the pkl file (with the extension .pkl), default: [name_of_the_json]_embedding.pkl')
    parser.add_argument('--max_tokens', type=int, default=8191, 
                       help='Maximum tokens for embedding model')
    parser.add_argument('--type_data_json', type=bool, default=False, 
                       help='in a text or in a dictionary format, False if in text format')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create embedding model. Using the OpenAI embedding model as an example.
    model = EnvironmentEmbeddingModel()
    
    # Create and initialize the EmbeddingDB asynchronously
    pkl_path = f"rag_documents/pkl_db/{args.pkl_name}" or f"rag_documents/pkl_db/{os.path.splitext(args.data_path)[0]}_embeddings.pkl"
    db = await CreateEmbeddingDB.create(
        args.data_path, 
        args.embedding_key, 
        model,
        pkl_path,
        args.type_data_json
    )
    
    # Now db is fully initialized and ready to use
    print(f"Database created with {len(db.embedding_index)} embeddings from {args.data_path}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

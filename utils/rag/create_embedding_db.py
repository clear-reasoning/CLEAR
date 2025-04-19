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
    async def create(cls, file_path: str, embedding_model, pkl_path: str = None):
        """Async factory method to create and initialize the EmbeddingDB"""
        instance = cls(file_path, embedding_model, pkl_path)
        await instance._initialize()
        return instance

    def __init__(self, json_file_path: str, embedding_model, pkl_path: str = None):
        # this is a JSON file path that we will read to create our "embedding_index"

        self.file_path = json_file_path
        self.embedding_index: np.ndarray = None
        self.db_rows = None
        self.pkl_path = pkl_path

        with open(json_file_path, "r") as f:
            self.json_contents = json.load(f)

        # Saving the embedding model
        self.embedding_model = embedding_model

    async def _initialize(self):
        """
        Async initialization to store the timesteps in a dataframe and the token embeddings of the explanation text.
        in the pkl file, in self.db_rows for each doc we will have:
        - current_situation: dict -> {"speed": 30, "headway": 25, "leader_speed": 20}
        - embed_current_situation: embedding of the current situation (which as a text would be: "speed: 30, headway: 25, leader_speed: 20")
        - action_analysis: embedding of the action analysis
        - situation_analysis: embedding of the situation analysis
        - hypothetical_situation: embedding of the hypothetical situation
        - ground_truth_explanation: embedding of the ground truth explanation
        - embedding_model: "OpenAIEmbeddingModel" if self.embedding_model is OpenAIEmbeddingModel, else "EnvironmentEmbeddingModel"
        """
        embeddings = [
            []
        ]  # list of list of embeddings for each row and for each key to embed
        for row in self.json_contents:
            text_current_situation = ", ".join(
                f"{key}: {value}" for key, value in row["current_situation"].items()
            )
            contents_to_vectorize = [
                text_current_situation,
                row["action_analysis"],
                row["situation_analysis"],
                row["hypothetical_situation"],
                row["ground_truth_explanation"],
            ]
            # Asyncly get the embeddings for the contents
            row_embeddings = await self.embedding_model.get_embeddings_batch_async(
                contents_to_vectorize
            )
            embeddings.append(np.array([emb for emb, _ in row_embeddings]))

        # Store the embeddings
        self.embedding_index = np.array([embeddings for embeddings in embeddings])

        data_to_store = []
        for i, row in enumerate(self.json_contents):
            # Add the embedding to the data to store
            data_to_store.append(
                {
                    "current_situation": row["current_situation"],
                    "embed_current_situation": self.embedding_index[i][0],
                    "action_analysis": self.embedding_index[i][1],
                    "situation_analysis": self.embedding_index[i][2],
                    "hypothetical_situation": self.embedding_index[i][3],
                    "ground_truth_explanation": self.embedding_index[i][4],
                    "embedding_model": row["embedding_model"],
                }
            )

        # Store the data
        self.db_rows = data_to_store

        # Save to pickle file
        self.save_to_pickle()
        print(f"Embeddings saved to {self.pkl_path}")

    def save_to_pickle(self):
        """Save the necessary data to a pickle file (excluding the model)"""
        data_to_save = {
            "embedding_index": self.embedding_index,
            "file_path": self.file_path,
            "db_rows": self.db_rows,
            "json_contents": self.json_contents,
        }
        with open(self.pkl_path, "wb") as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load_from_pickle(cls, pkl_path: str, embedding_model):
        """Load data from pickle and create a new EmbeddingDB instance"""
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Create a new instance
        instance = cls(data["file_path"], embedding_model, pkl_path)

        # Set the loaded data
        instance.embedding_index = data["embedding_index"]

        return instance

    def get_top_k_docs(
        self,
        hypothetical_situation: str,
        current_situation: dict,
        first_k: int = 30,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get the top K documents based on the hypothetical situation and the current situation.
        Args:
        - hypothetical_situation: The hypothetical situation of the new doc we are creating and for which we want to find similar situations.
        - current_situation: The current situation dictionary. {"speed": 30, "headway": 25, "leader_speed": 20}
        - first_k: The intermediate number of top documents to consider for the hypothetical situation.
        - k: The finale number of top documents to return (based on the current situation).
        """
        # Get the embedding for the hypothetical situation provided by the user
        embedding_user_hyp_sit = self.embedding_model.get_embedding(
            hypothetical_situation
        )

        # Calculate cosine similarity
        embeddings_hypothetical_situation = self.embedding_index[
            3
        ]  # list of embeddings for the hypothetical situation of the rows in the rag db
        similarities = np.dot(
            embeddings_hypothetical_situation, embedding_user_hyp_sit
        ) / (
            np.linalg.norm(embeddings_hypothetical_situation, axis=1)
            * np.linalg.norm(embedding_user_hyp_sit)
        )

        # Get the top first_k most similar hypothetical situations
        top_first_k_indices = np.argsort(similarities)[-first_k:]

        first_k_docs = [self.json_contents[i] for i in top_first_k_indices]

        # Get the top K most similar current situations from the first_k_docs
        l2_loss = []
        for doc in first_k_docs:
            l2_loss.append(
                (doc["current_situation"]["speed"] - current_situation["speed"]) ** 2
                + (doc["current_situation"]["headway"] - current_situation["headway"])
                ** 2
                + (
                    doc["current_situation"]["leader_speed"]
                    - current_situation["leader_speed"]
                )
                ** 2
            )

        top_k_indices = np.argsort(l2_loss)[-k:]

        # Return the top K documents and their corresponding texts
        return [first_k_docs[i] for i in top_k_indices]


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create embeddings database from JSON data"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the JSON data file"
    )
    parser.add_argument(
        "--pkl_name",
        type=str,
        default=None,
        help="Name of the pkl file (with the extension .pkl), default: [name_of_the_json]_embedding.pkl",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8191,
        help="Maximum tokens for embedding model",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create embedding model. Using the OpenAI embedding model as an example.
    model = EnvironmentEmbeddingModel()

    # Create and initialize the EmbeddingDB asynchronously
    pkl_path = (
        f"rag_documents/pkl_db/{args.pkl_name}"
        or f"rag_documents/pkl_db/{os.path.splitext(args.data_path)[0]}_embeddings.pkl"
    )
    db = await CreateEmbeddingDB.create(
        args.data_path,
        model,
        pkl_path,
    )

    # Now db is fully initialized and ready to use
    print(
        f"Database created with {len(db.embedding_index)} embeddings from {args.data_path}"
    )


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

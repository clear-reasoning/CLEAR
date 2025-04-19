import pickle
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Callable, Any, Dict
from sentence_transformers import SentenceTransformer, util
from utils.rag.embedding_models import (
    BaseEmbeddingModel,
    OpenAIEmbeddingModel,
    EnvironmentEmbeddingModel,
)
from utils.rag.similarity_functions import (
    cosine_similarity,
    l2_loss,
    compare_dataframes,
)
from utils.df_utils import get_formatted_df_for_llm
import os


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

    def __init__(self, db_pkl_path: str = None, db_max_size: int = None):
        self.db_pkl_path = db_pkl_path
        self.db_data = None
        self.db_max_size = db_max_size

        if db_pkl_path is None:
            self._initialize_empty_db()
        else:
            with open(db_pkl_path, "rb") as f:
                self.db_data = pickle.load(f)

        # Retrieving the content of the pkl file
        self.embedding_index: list = self.db_data["embedding_index"]
        self.file_path: str = self.db_data["file_path"]
        self.db_rows = self.db_data.get(
            "db_rows", []
        )  # Note: db_rows is not always present in the pickle file.
        self.json_contents = self.db_data.get(
            "json_contents", []
        )  # Note: json_contents is not always present in the pickle file.

    def _initialize_empty_db(self):
        """Initialize an empty database with default values"""
        self.db_data = {
            "embedding_index": [],
            "file_path": None,
            "db_rows": [],
            "json_contents": [],
        }

        print("Initialized empty database in memory")

    def replace_document(
        self, document_index: int, new_json_content: Dict[str, Any], embedding_model
    ) -> None:
        """Replace a document in the database

        Args:
            document_index (int): The index of the document to replace
            new_json_content (Dict[str, Any]): The new content for the document
            embedding_model (BaseEmbeddingModel): The embedding model to use
        """
        assert document_index < len(
            self.embedding_index
        ), "Document index is out of range"
        assert isinstance(
            new_json_content, dict
        ), "Document contents must be a dictionary"

        text_current_situation = ", ".join(
            f"{key}: {value}"
            for key, value in new_json_content["current_situation"].items()
        )

        if embedding_model is None:
            raise ValueError("No embedding model specified in the new JSON content")
        embedding_model_to_store = (
            "OpenAIEmbeddingModel"
            if isinstance(embedding_model, OpenAIEmbeddingModel)
            else "EnvironmentEmbeddingModel"
        )

        new_data_to_store = {
            "current_situation": new_json_content["current_situation"],
            "embed_current_situation": embedding_model.get_embedding(
                text_current_situation
            ),
            "action_explanation": embedding_model.get_embedding(
                new_json_content["action_explanation"]
            ),
            "action": new_json_content["action"],
            "hypothetical_situation": embedding_model.get_embedding(
                new_json_content["hypothetical_situation"]
            ),
            "situation_analysis": embedding_model.get_embedding(
                new_json_content["situation_analysis"]
            ),
            "embedding_model": embedding_model_to_store,
        }

        self.embedding_index[document_index] = [
            np.array(new_data_to_store["embed_current_situation"]),
            np.array(new_data_to_store["action_analysis"]),
            np.array(new_data_to_store["hypothetical_situation"]),
            np.array(new_data_to_store["situation_analysis"]),
        ]

        self.db_rows[document_index] = new_data_to_store
        self.json_contents[document_index] = new_json_content

        self.db_data = {
            "embedding_index": self.embedding_index,
            "file_path": self.file_path,
            "db_rows": self.db_rows,
            "json_contents": self.json_contents,
        }

    def add_document(self, new_json_content: Dict[str, Any], embedding_model) -> None:
        """Add a document to the database

        Args:
            new_json_content (Dict[str, Any]): The content for the new document
            embedding_model (BaseEmbeddingModel): The embedding model to use
        """
        assert isinstance(
            new_json_content, dict
        ), "Document contents must be a dictionary"

        text_current_situation = ", ".join(
            f"{key}: {value}"
            for key, value in new_json_content["current_situation"].items()
        )
        embedding_model_to_store = (
            "OpenAIEmbeddingModel"
            if isinstance(embedding_model, OpenAIEmbeddingModel)
            else "EnvironmentEmbeddingModel"
        )

        new_data_to_store = {
            "current_situation": new_json_content["current_situation"],
            "embed_current_situation": embedding_model.get_embedding(
                text_current_situation
            ),
            "action_explanation": embedding_model.get_embedding(
                new_json_content["action_explanation"]
            ),
            "action": new_json_content["action"],
            "hypothetical_situation": embedding_model.get_embedding(
                new_json_content["hypothetical_situation"]
            ),
            "situation_analysis": embedding_model.get_embedding(
                new_json_content["situation_analysis"]
            ),
            "embedding_model": embedding_model_to_store,
        }

        new_embedding = [
            np.array(new_data_to_store["embed_current_situation"]),
            np.array(new_data_to_store["action_explanation"]),
            np.array(new_data_to_store["hypothetical_situation"]),
            np.array(new_data_to_store["situation_analysis"]),
        ]

        if len(self.embedding_index) == 0:
            self.embedding_index = [new_embedding]
        else:
            embedding_index = self.embedding_index
            embedding_index.append(new_embedding)
            self.embedding_index = embedding_index

        self.db_rows.append(new_data_to_store)
        self.json_contents.append(new_json_content)

        self.db_data = {
            "embedding_index": self.embedding_index,
            "file_path": self.file_path,
            "db_rows": self.db_rows,
            "json_contents": self.json_contents,
        }

    def remove_document(self, document_index: int) -> None:
        """Remove a document from the database

        Args:
            document_index (int): The index of the document to remove
        """
        self.embedding_index.pop(document_index)
        self.db_rows.pop(document_index)
        self.json_contents.pop(document_index)

        self.db_data = {
            "embedding_index": self.embedding_index,
            "file_path": self.file_path,
            "db_rows": self.db_rows,
            "json_contents": self.json_contents,
        }

    def save_database(self, save_path: str = None) -> None:
        """Save the database to a pickle file

        Args:
            save_path (str, optional): The path to save the database. If None, uses the original db_pkl_path
        """
        if save_path is None:
            save_path = self.db_pkl_path

        # Update db_data with current state before saving
        self.db_data = {
            "embedding_index": self.embedding_index,
            "file_path": self.file_path,
            "db_rows": self.db_rows,
            "json_contents": self.json_contents,
        }

        with open(save_path, "wb") as f:
            pickle.dump(self.db_data, f)

        print(f"Database saved successfully to {save_path}")

    def visualize_database_as_json(self, json_name: str = None) -> None:
        """Visualize the database as a JSON file
        Args:
            json_name (str): The name of the JSON file to save the database in the rag_documents/json directory
        """
        with open(self.db_pkl_path, "rb") as f:
            data = pickle.load(f)

        data_to_print = data["json_contents"]

        # self.db_pkl_path in rag_documents/pkl_db directory
        # save the database as a JSON file in the rag_documents/json directory
        json_dir = os.path.join(os.path.dirname(self.db_pkl_path), "json")
        os.makedirs(json_dir, exist_ok=True)
        if json_name is None:
            # if no name is provided, use the name of the pickle file
            json_name = os.path.basename(self.db_pkl_path).replace(".pkl", ".json")
        json_path = os.path.join(json_dir, json_name)

        with open(json_path, "w") as f:
            json.dump(data_to_print, f, indent=4, sort_keys=True)

        print(f"Database can be visualized in the JSON file in {json_path}")

    def get_doc_from_index(self, index: int) -> Dict[str, Any]:
        """Get a document from the database by index

        Args:
            index (int): The index of the document to retrieve

        Returns:
            Dict[str, Any]: The document at the specified index
        """
        if index < 0 or index >= len(self.json_contents):
            raise IndexError("Index out of range")
        return self.json_contents[index]

    def get_size(self) -> int:
        """Get the size of the database

        Returns:
            int: The number of documents in the database
        """
        return len(self.json_contents)

    def get_top_k_docs(
        self,
        hypothetical_situation: str,
        current_situation: dict,
        embedding_model,
        intermediate_k: int = 30,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get the top K documents based on the hypothetical situation and the current situation.
        Args:
        - user_hypothetihypothetical_situationcal_situation: The hypothetical situation for the document we are creating and for which we want to find similar documents.
        - current_situation: The current situation of the vehicle.
        - intermediate_k: The number of intermediate documents to consider (documents most similar based on the hypothetical situation).
        - k: The number of top documents to return (among intermediate documents, most similar ones based on the current situation).
        """

        # NOTE: similarity between the embeddings doesn't work because the function get_embedding returns vectors of different sizes.
        # # Get the embedding for the hypothetical situation provided by the user
        # embedding_new_doc_hyp_sit = embedding_model.get_embedding(
        #     hypothetical_situation
        # )

        # # Calculate cosine similarity
        # embeddings_hypothetical_situation = self.db_data["embedding_index"][3]
        # similarities = np.dot(
        #     embeddings_hypothetical_situation, embedding_new_doc_hyp_sit
        # ) / (
        #     np.linalg.norm(embeddings_hypothetical_situation)
        #     * np.linalg.norm(embedding_new_doc_hyp_sit)
        # )

        # NOTE: I compute the similarity without using the embedding stored in the rag db
        # Load pre-trained model
        similarities = []
        model = SentenceTransformer("all-MiniLM-L6-v2")
        for i in range(len(self.json_contents)):
            text1 = self.db_data["json_contents"][i]["hypothetical_situation"]
            text2 = hypothetical_situation

            # Compute embeddings for both texts
            embedding1 = model.encode(text1, convert_to_tensor=True)
            embedding2 = model.encode(text2, convert_to_tensor=True)

            cosine_similarity = util.cos_sim(embedding1, embedding2)
            cosine_similarity_score = cosine_similarity.item()

            similarities.append(cosine_similarity_score)

        # Get the top first_k most similar hypothetical situations
        similarities = np.array(similarities)
        top_intermediate_k_indices = np.argsort(similarities)[-intermediate_k:]

        intermediate_k_docs = []
        for i in top_intermediate_k_indices:
            doc = self.json_contents[i]
            doc["original_index"] = i  # Add the original index
            intermediate_k_docs.append(doc)

        # Get the top K most similar current situations from the first_k_docs
        l2_loss = []
        for doc in intermediate_k_docs:
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
        return [intermediate_k_docs[i] for i in top_k_indices]

    def save_as_json(self, save_path: str = None):
        """Save the database as a JSON file

        Args:
            save_path (str, optional): The path to save the JSON file. If None, uses the original db_pkl_path
        """
        if save_path is None:
            save_path = self.db_pkl_path.replace(".pkl", ".json")

        # only saving self.json_contents as the JSON file
        with open(save_path, "w") as f:
            json.dump(self.json_contents, f, indent=4, sort_keys=True)

    def add_entry(self, embedding: np.ndarray, json_content: Dict[str, Any]):
        """Add a new entry to the database"""
        # Update embedding index
        if len(self.embedding_index) == 0:
            self.embedding_index = embedding.reshape(1, -1)
        else:
            self.embedding_index = np.vstack([self.embedding_index, embedding])

        # Update json contents
        self.json_contents.append(json_content)

        # Update db_data
        self.db_data["embedding_index"] = self.embedding_index
        self.db_data["json_contents"] = self.json_contents

        # Save updated database
        with open(self.db_pkl_path, "wb") as f:
            pickle.dump(self.db_data, f)

    def get_similar_entries(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the most similar entries to the query embedding"""
        if len(self.embedding_index) == 0:
            return []

        # Calculate cosine similarity
        similarities = np.dot(self.embedding_index, query_embedding) / (
            np.linalg.norm(self.embedding_index, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return corresponding json contents
        return [self.json_contents[i] for i in top_k_indices]


if __name__ == "__main__":
    # Example usage of using the RAG_Database
    db = RAG_Database("environment_db_examples.pkl")
    embedding_model = EnvironmentEmbeddingModel()

    # Adding a new document to the database
    db.add_document(
        {
            "current_situation": {"speed": 71.0, "headway": 35.0, "leader_speed": 70.5},
            "action_explanation": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <action>0.5<\/action>",
            "action": 0.5,
            "hypothetical_situation": "What happens if the headway suddenly decreases from 18m to 10m over the next 3 seconds?",
            "situation_analysis": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <prediction>The vehicle will brake in 2 seconds.<\/prediction>",
            "embedding_model": "OpenAIEmbeddingModel()",
        },
        embedding_model=embedding_model,
    )

    top_k_docs = db.get_top_k_docs(
        hypothetical_situation="What would happen if the car in front slows down and then cuts out of the lane?",
        current_situation={"speed": 62.0, "headway": 25.0, "leader_speed": 70.5},
        intermediate_k=30,
        k=5,
    )
    for doc in top_k_docs:
        print(doc)

    # replace the first document with a new one
    db.replace_document(
        0,
        {
            "current_situation": {"speed": 71.0, "headway": 35.0, "leader_speed": 70.5},
            "action_explanation": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <action>0.5<\/action>",
            "action": 0.5,
            "hypothetical_situation": "What happens if the headway suddenly decreases from 18m to 10m over the next 3 seconds?",
            "situation_analysis": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <prediction>The vehicle will brake in 2 seconds.<\/prediction>",
            "embedding_model": "OpenAIEmbeddingModel()",
        },
        embedding_model=embedding_model,
    )

    print("Replaced the first document")

    top_k_docs = db.get_top_k_docs(
        hypothetical_situation="What would happen if the car in front slows down and then cuts out of the lane?",
        current_situation={"speed": 62.0, "headway": 25.0, "leader_speed": 70.5},
        intermediate_k=30,
        k=5,
    )
    for doc in top_k_docs:
        print(doc)

    # adding more documents with the same key before saving the database
    db.add_document(
        {
            "current_situation": {"speed": 71.0, "headway": 35.0, "leader_speed": 70.5},
            "action_explanation": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <action>0.5<\/action>",
            "action": 0.5,
            "hypothetical_situation": "What happens if the headway suddenly decreases from 18m to 10m over the next 3 seconds?",
            "situation_analysis": "<observation>First observation<\/observation> <deduction>Second observation<\/deduction> <prediction>The vehicle will brake in 2 seconds.<\/prediction>",
            "embedding_model": "OpenAIEmbeddingModel()",
        },
        embedding_model=embedding_model,
    )

    db.save_database("environment_db_examples_new.pkl")
    db = RAG_Database("environment_db_examples_new.pkl")

    db.remove_document(0)  # remove the first document

    # remove the first document
    print("Removed the first document")

    top_k_docs = db.get_top_k_docs(
        hypothetical_situation="What would happen if the car in front slows down and then cuts out of the lane?",
        current_situation={"speed": 62.0, "headway": 25.0, "leader_speed": 70.5},
        intermediate_k=30,
        k=5,
    )
    for doc in top_k_docs:
        print(doc)

    # save the database as a JSON file
    db.save_as_json("environment_db_examples_new.json")

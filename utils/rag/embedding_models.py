import os
import asyncio
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken
from openai import OpenAI
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models with chunking support"""
    def __init__(self, max_tokens: int = 512, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text input"""
        pass
    
    async def get_embedding_async(self, text: str) -> List[float]:
        """Async wrapper for get_embedding method"""
        # Run synchronous get_embedding in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)
    
    def validate_embedding_size(self, text: str) -> bool:
        """Check if the text exceeds the max token limit"""
        if self.tokenizer is None:
            return True
        num_tokens = len(self.tokenizer.encode(text))
        if num_tokens > self.max_tokens:
            return False
        return True
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, str]]:
        """Batch process embeddings with text pairing"""
        return [(self.get_embedding(text), text) for text in texts]
    
    async def get_embeddings_batch_async(self, texts: List[str]) -> List[Tuple[np.ndarray, str]]:
        """Async batch process embeddings with text pairing"""
        # Create tasks for all texts
        tasks = [self.get_embedding_async(text) for text in texts]
        
        # Execute all tasks concurrently
        embeddings = await asyncio.gather(*tasks)
        
        # Return tuples of (embedding, text)
        return [(embedding, text) for embedding, text in zip(embeddings, texts)]

class MiniLMEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_tokens = 256  # Model's actual max sequence length
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Return embedding with original text"""
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Tuple[np.ndarray, str]]:
        """Optimized batch processing with text pairing"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-3-small", max_tokens=8191):
        super().__init__()
        import tiktoken
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens  # OpenAI's limit
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Return OpenAI embedding with original text"""
        # Truncation logic remains the same
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

class EnvironmentEmbeddingModel(BaseEmbeddingModel):        
    def get_embedding(self, text: str) -> np.ndarray:
        """Convert the text representation of the environment into a vector"""
        import re
        import numpy as np
        
        values = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
        
        embedding = np.array([float(value) for value in values])
        
        return embedding
    
if __name__ == "__main__":
    # Usage example
    async def main():
        model = OpenAIEmbeddingModel()
        texts = ["sample text 1", "sample text 2", "sample text 3"]
        
        # Async batch processing
        results = await model.get_embeddings_batch_async(texts)
        for embedding, text in results:
            print(f"Text: {text[:10]}..., Embedding shape: {len(embedding)}")

    asyncio.run(main())
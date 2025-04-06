import pandas as pd
from typing import List

class TrajectoryChunker:
    def __init__(self, trajectory_df: pd.DataFrame, chunk_col: str, chunk_indices: List[int], sort_col: str | None = None):
        """
        Initializes the TrajectoryChunker with a DataFrame and chunk size.
        
        :param trajectory_df: DataFrame containing trajectory data.
        :param chunk_size: Size of each chunk.
        """
        # Ensure that the type of the chunk_col in the dataframe is either int or float
        if not pd.api.types.is_numeric_dtype(trajectory_df[chunk_col]):
            raise ValueError(f"Column {chunk_col} must be numeric.")

        self.trajectory_df = trajectory_df
        
        # storing the column to chunk by
        self.chunk_col = chunk_col
        self.sort_col = sort_col
        
        # storing the actual indices that will be segment the trajectory based on
        self.chunk_indices = chunk_indices
        
        # adding the minimum and maximum values to the chunk indices if they are not present
        if self.chunk_indices[0] != trajectory_df[chunk_col].min():
            self.chunk_indices = [trajectory_df[chunk_col].min()] + self.chunk_indices
        if self.chunk_indices[-1] != trajectory_df[chunk_col].max():
            self.chunk_indices = self.chunk_indices + [trajectory_df[chunk_col].max()]
        
        if sort_col:
            self.trajectory_df = self.trajectory_df.sort_values(by=sort_col)
    
    def get_chunk(self, chunk_index: int) -> pd.DataFrame:
        """
        Returns a chunk of the trajectory DataFrame based on the specified chunk index.
        
        :param chunk_index: Index of the chunk to retrieve.
        :return: DataFrame containing the specified chunk.
        """
        if chunk_index < 0 or chunk_index >= len(self.chunk_indices):
            raise ValueError("Chunk index out of range.")
        
        slice_begin_index = self.chunk_indices[chunk_index]
        slice_end_index = self.chunk_indices[chunk_index + 1]
        
        return self.trajectory_df.iloc[slice_begin_index:slice_end_index]
    
    def get_chunks(self) -> List[pd.DataFrame]:
        """
        Returns a list of all chunks of the trajectory DataFrame.
        
        :return: List of DataFrames, each representing a chunk.
        """
        chunks = []
        for i in range(len(self.chunk_indices) - 1):
            chunks.append(self.get_chunk(i))
        return chunks
    
    def get_num_chunks(self) -> int:
        """
        Returns the number of chunks.
        
        :return: Number of chunks.
        """
        return len(self.chunk_indices) - 1

class TrajectoryWindows:
    def __init__(self, trajectory: pd.DataFrame | pd.DataFrame, window_size: int, indexing_col: str):
        """
        Initializes the TrajectoryWindows with a trajectory and window size.
        
        :param trajectory: Trajectory data (either TrajectoryChunks or DataFrame).
        :param window_size: Size of each window.
        :param step_size: Step size for moving the window.
        """
        self.indexing_col = indexing_col
        self.trajectory = trajectory
        self.window_size = window_size
    
    def get_windows(self) -> List[pd.DataFrame]:
            """
            Returns a list of DataFrames, each representing a sliding window of the trajectory DataFrame.
            
            :return: List of DataFrames, each representing a window of size window_size.
            """
            # Get the number of windows (excluding the last part if not enough rows for a full window)
            num_rows = len(self.trajectory)
            windows = [
                self.trajectory.iloc[i:i + self.window_size] 
                for i in range(num_rows - self.window_size + 1)
            ]
            return windows
        
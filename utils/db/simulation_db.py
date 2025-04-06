import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

class SimulationDB:
    def __init__(self, db_path: Union[str, Path]):
        """Initialize database connection handler.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn = None
        
    def connect(self):
        """Establish connection to the database."""
        if not self._conn:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn
    
    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_column_names(self, table_name: str = "timestep_data") -> List[str]:
        """Get column names for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        return columns
    
    def get_tables(self) -> List[str]:
        """Get all table names in the database.
        
        Returns:
            List of table names
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        return tables
    
    def query_to_df(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        conn = self.connect()
        return pd.read_sql_query(query, conn)
    
    def get_timestep_data(self, limit: Optional[int] = None, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Get timestep data as a DataFrame.
        
        Args:
            limit: Optional limit on number of rows
            columns: Optional list of specific columns to retrieve
            
        Returns:
            DataFrame with timestep data
        """
        if columns is None:
            columns_str = "*"
        else:
            columns_str = ", ".join([f'"{col}"' for col in columns])
            
        limit_str = f"LIMIT {limit}" if limit else ""
        
        query = f"SELECT {columns_str} FROM timestep_data {limit_str}"
        return self.query_to_df(query)
    
    def get_vehicle_data(self, vehicle_id: str, 
                         data_keys: Optional[List[str]] = None) -> pd.DataFrame:
        """Get data for a specific vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            data_keys: Optional list of specific data keys to retrieve
            
        Returns:
            DataFrame with vehicle data
        """
        all_columns = self.get_column_names()
        
        # Get timestep column and all columns for this vehicle
        vehicle_columns = ["timestep", "reward"] + [
            col for col in all_columns 
            if col.startswith(f"{vehicle_id}__")
        ]
        
        # Filter by data keys if specified
        if data_keys:
            vehicle_columns = ["timestep", "reward"] + [
                col for col in vehicle_columns[1:]
                if any(col.endswith(f"__{key}") for key in data_keys)
            ]
        
        return self.get_timestep_data(columns=vehicle_columns)
    
    def get_all_vehicle_ids(self) -> List[str]:
        """Extract all unique vehicle IDs from column names.
        
        Returns:
            List of vehicle IDs
        """
        columns = self.get_column_names()
        vehicle_ids = set()
        
        for col in columns:
            if "__" in col:
                vehicle_id = col.split("__")[0]
                vehicle_ids.add(vehicle_id)
                
        return list(vehicle_ids)
    
    def get_data_keys(self) -> List[str]:
        """Extract all unique data keys from column names.
        
        Returns:
            List of data keys
        """
        columns = self.get_column_names()
        data_keys = set()
        
        for col in columns:
            if "__" in col:
                data_key = col.split("__")[1]
                data_keys.add(data_key)
                
        return list(data_keys)
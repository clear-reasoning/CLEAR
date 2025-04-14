import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter
from utils.trajectory.trajectory_container import TrajectoryChunker
from typing import List, Optional

def get_df_from_csv(csv_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file_path, na_values=["NULL", "NaN", "N/A", ""])
    return df

def convert_cols_to_numeric(df: pd.DataFrame, cols: list[str], null_key: str = "NULL") -> pd.DataFrame:
    """
    Converts specified columns in a DataFrame to numeric, replacing NULL values with NaN.
    
    :param df: Pandas DataFrame to be processed.
    :param cols: List of column names to convert.
    :param null_key: Value to be replaced with NaN (default is "NULL").
    :return: DataFrame with specified columns converted to numeric.
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col].replace(null_key, np.nan), errors='coerce')
    return df

def plot_dataframe(df: pd.DataFrame, 
                   x_axis: str, 
                   y_values: list[str], 
                   save_path: str, 
                   epsilon: float = 0.7, 
                   max_ticks: int = 10, 
                   chunker: Optional[TrajectoryChunker] = None, 
                   shaded_regions: List[tuple] = []) -> None:
    """
    Plots a line chart for the given DataFrame. For the 'speed' column, if 'accel' is present,
    encodes acceleration as color. Other y-values are plotted normally.
    """
    # Clean and convert data
    df = df.copy()
    df[x_axis] = pd.to_numeric(df[x_axis].fillna(-1).replace("NULL", 0), errors='coerce').fillna(0)
    df[y_values] = df[y_values].apply(lambda col: pd.to_numeric(col.fillna(-1).replace("NULL", 0), errors='coerce').fillna(0))

    # Include accel if available
    has_accel = 'accel' in df.columns
    if has_accel:
        df['accel'] = pd.to_numeric(df['accel'].fillna(0).replace("NULL", 0), errors='coerce').fillna(0)

    # Sampling
    if 0.0 < epsilon < 1.0:
        df = df.sample(frac=epsilon, random_state=42).sort_values(by=x_axis)

    plt.figure(figsize=(18, 6))

    for y in y_values:
        if y == 'speed' and has_accel:
            sc = plt.scatter(df[x_axis], df[y], c=df['accel'], cmap='coolwarm', s=0.5, label=f'{y} (colored by accel)')
            cbar = plt.colorbar(sc)
            cbar.set_label("Acceleration (m/s²)")
        else:
            plt.scatter(df[x_axis], df[y], marker='X', label=y, s=0.25)

    # Shaded regions
    for xmin, xmax, color, alpha in shaded_regions:
        plt.axvspan(xmin, xmax, color=color, alpha=alpha)

    # Chunker boundaries
    if chunker is not None:
        for chunk_index in chunker.chunk_indices:
            boundary_x = df[df[chunker.chunk_col] == chunk_index][x_axis].min()
            plt.axvline(x=boundary_x, color='red', linestyle='dotted', linewidth=2)

    plt.xlabel(x_axis)
    plt.ylabel("Values")
    plt.title("Trajectory Plot with Optional Acceleration Encoding")
    plt.grid(True)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

    plt.savefig(save_path)
    plt.close()

def get_formatted_df_for_llm(df, max_rows=None, precision=4):
    """
    Formats a pandas DataFrame into a clean, readable text format for LLMs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to format
    max_rows : int, optional
        Maximum number of rows to include. If None, includes all rows
    precision : int, optional
        Number of decimal places to round floating point numbers to
    
    Returns:
    --------
    str
        A formatted string representation of the DataFrame
    """
    # Make a copy to avoid modifying the original
    df_display = df.copy()
    
    # Limit rows if specified
    if max_rows is not None and len(df_display) > max_rows:
        half = max_rows // 2
        df_display = pd.concat([df_display.head(half), df_display.tail(half)])
    
    # Round floating point numbers
    for col in df_display.select_dtypes(include=['float']).columns:
        df_display[col] = df_display[col].round(precision)
    
    # Convert to string with clean formatting
    result = "DataFrame with shape: {} rows × {} columns\n\n".format(*df_display.shape)
    
    # Add column descriptions
    result += "Columns:\n"
    for col in df_display.columns:
        dtype = str(df_display[col].dtype)
        result += f"- {col} ({dtype})\n"
    
    result += "\nData:\n"
    
    # Format as markdown table
    header = "| " + " | ".join(str(col) for col in df_display.columns) + " |"
    separator = "| " + " | ".join("-" * len(str(col)) for col in df_display.columns) + " |"
    
    result += header + "\n" + separator + "\n"
    
    # Add rows
    for _, row in df_display.iterrows():
        result += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    
    return result
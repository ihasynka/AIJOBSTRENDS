import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

def plot_bar_chart(data_series: Union[pd.Series, pd.DataFrame], 
                   title: str, 
                   xlabel: str, 
                   ylabel: str, 
                   save_path: str = None) -> None:
    """
    Generates and displays a bar chart for a given pandas Series or DataFrame column.

    This function is used by AITrendsAnalyzer to visualize salary statistics and 
    technology popularity.

    Args:
        data_series: A pandas Series (e.g., skill counts or average salaries) 
                     or a DataFrame where the first column is the data to plot.
        title: The title of the chart.
        xlabel: The label for the X-axis (categories).
        ylabel: The label for the Y-axis (values).
        save_path: Optional path to save the chart (e.g., 'chart.png'). 
                   If None, the chart is displayed.
    
    Raises:
        TypeError: If data_series is not a pandas Series or DataFrame.
    """
    
    if not isinstance(data_series, (pd.Series, pd.DataFrame)):
        raise TypeError("data_series must be a pandas Series or DataFrame.")

    if isinstance(data_series, pd.DataFrame):
        data = data_series.iloc[:, 0]
    else:
        data = data_series
        
    plt.figure(figsize=(12, 6))
    
    data.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Chart saved to {save_path}")
        except Exception as e:
            print(f"Error saving chart: {e}")
    else:
        plt.show()

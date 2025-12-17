import pandas as pd
import numpy as np
import re
from typing import Union, List, Dict
from pathlib import Path
from collections import Counter
import warnings
from aijobstrends.visualization.plotter import plot_bar_chart
try:
    from aijobstrends.visualization.plotter import plot_bar_chart 
except ImportError:
    # Fallback if the plotting dependency or file is missing
    def plot_bar_chart(*args, **kwargs):
        warnings.warn("plot_bar_chart is not available. Check aijobstrends/visualization/plotter.py")


class AITrendsAnalyzer:
    """
    Main class for analyzing AI job market trends.

    The class encapsulates the workflow: data loading, cleaning, analysis (salary, 
    technology popularity), and visualization.
    
    Attributes:
        file_path (Path): Path to the loaded CSV file.
        role_col (str): The name of the column containing job roles ('job_title').
        salary_col (str): The name of the column containing average salary ('salary_in_usd').
        skills_col (str): The name of the column containing required skills ('skills_required').
        data (pd.DataFrame): Cleaned and analysis-ready data.
    """

    def __init__(self, file_path: Union[str, Path], 
                 role_col: str = 'job_title',
                 salary_col: str = 'salary_in_usd', 
                 skills_col: str = 'skills_required'):
        """
        Initializes the analyzer by loading data and performing basic validation.

        Args:
            file_path: Path to the CSV file containing job data.
            role_col: Name of the column with job roles (default 'job_title').
            salary_col: Internal name for the column with average salary (default 'salary_in_usd').
            skills_col: Name of the column listing technologies (default 'skills_required').
        
        Raises:
            FileNotFoundError: If the file is not found at the specified path.
            ValueError: If the CSV lacks required columns after cleaning.
            TypeError: If input arguments have an incorrect type.
        """
        if not isinstance(file_path, (str, Path)):
             raise TypeError("file_path must be a string or Path object.")
             
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found at path: {file_path}")

        self.file_path = path
        self.role_col = role_col
        self.salary_col = salary_col
        self.skills_col = skills_col
        
        self.data: pd.DataFrame = self._load_and_clean_data()
        
        required_cols = [self.role_col, self.salary_col, self.skills_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(
                f"Data is missing essential columns after processing: {missing_cols}. "
                f"Please ensure the input CSV contains 'job_title' and 'salary_range_usd'."
            )


    def _load_and_clean_data(self) -> pd.DataFrame:
        """
        Loads the data, converts 'salary_range_usd' into a numerical average 
        ('salary_in_usd'), and cleans the DataFrame.
        """
        df = pd.read_csv(self.file_path)
        
        if 'Unnamed: 0' in df.columns or df.columns[0] == 'job_id':
            # Remove the index column to avoid conflicts with 'job_title'
            df.drop(columns=[df.columns[0]], inplace=True, errors='ignore')

        ORIGINAL_SALARY_COL = 'salary_range_usd'
        if ORIGINAL_SALARY_COL in df.columns:
            
            def calculate_mean_salary(salary_range: str) -> float:
                if pd.isna(salary_range):
                    return np.nan
                try:
                    low, high = map(float, salary_range.split('-'))
                    return (low + high) / 2
                except:
                    return np.nan

            df[self.salary_col] = df[ORIGINAL_SALARY_COL].apply(calculate_mean_salary)
    
            df.drop(columns=[ORIGINAL_SALARY_COL], inplace=True)
        
        if 'job_title' in df.columns:
            df.rename(columns={'job_title': self.role_col}, inplace=True)
     
        df.dropna(subset=[self.role_col, self.salary_col, self.skills_col], inplace=True)
        
        df[self.salary_col] = pd.to_numeric(df[self.salary_col], errors='coerce')

        return df


    def calculate_salary_stats(self) -> pd.DataFrame:
        """
        Calculates the average, median salary, and job count grouped by job role.
        
        Returns:
            pd.DataFrame: DataFrame with aggregated salary statistics.
        
        Example Usage:
        >>> analyzer = AITrendsAnalyzer('./aijobstrends/data/sample_data.csv')
        >>> stats = analyzer.calculate_salary_stats()
        >>> print(stats.head())
        """
        if self.data.empty:
            print("Warning: Data is empty.")
            return pd.DataFrame()
        
        stats = self.data.groupby(self.role_col)[self.salary_col].agg(
            average_salary='mean',
            median_salary='median',
            count='count'
        ).sort_values(by='count', ascending=False)
        
        plot_bar_chart(stats['average_salary'].head(10), 
                       f"Top 10 Average Salary by Job Role ({self.role_col})", 
                       self.role_col.capitalize(), 
                       "Average Salary (USD)")
                       
        return stats


    def get_technology_popularity(self, top_n: int = 10) -> pd.Series:
        """
        Counts the frequency of technologies/skills required for vacancies.

        Args:
            top_n: The number of top technologies to return (must be > 0).

        Returns:
            pd.Series: Series containing the technology name and its count.

        Raises:
            ValueError: If top_n is not a positive integer.

        Example Usage:
        >>> skills = analyzer.get_technology_popularity(5)
        >>> print(skills)
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
            
        if self.data.empty:
            return pd.Series(dtype='int64')

        def skill_generator(skills_series: pd.Series):
            """Generates cleaned individual skill names from a comma-separated Series."""
            for skills_str in skills_series.astype(str).dropna():
                for skill in skills_str.split(','):
                    clean_skill = skill.strip().lower()
                    if clean_skill and len(clean_skill) > 1:
                        yield clean_skill
        
        skill_counts = Counter(skill_generator(self.data[self.skills_col]))
             
        popularity = pd.Series(skill_counts).sort_values(ascending=False)
        
        plot_bar_chart(popularity.head(top_n), 
                       f"Top {top_n} Demanded AI Skills", 
                       "Skill", 
                       "Job Count")

        return popularity.head(top_n)


    def generate_report(self, top_n: int = 5) -> str:
        """
        Generates a concise textual report listing the "Top N demanded AI skills".

        Args:
            top_n: The number of top skills to include in the report (must be > 0).

        Returns:
            str: A formatted string containing the report.
        
        Example Usage:
        >>> report = analyzer.generate_report(3)
        >>> print(report)
        """
        try:
            top_skills = self.get_technology_popularity(top_n=top_n)
        except ValueError as e:
            return f"Error generating report: {e}"
        
        report = f"*** TOP {top_n} DEMANDED AI SKILLS REPORT ***\n\n"
        
        if top_skills.empty:
            report += "No skills data available for analysis."
            return report

        for rank, (skill, count) in enumerate(top_skills.items(), 1):
            report += f"{rank}. **{skill.capitalize()}**: {count} vacancies.\n"
        
        return report
import pandas as pd
from pathlib import Path
import re

class LeetcodeDataset:
    def __init__(self):
        # load dataset
        dataset_path = Path("./leetcode_dataset.csv")

        if dataset_path.exists():
            self._df = pd.read_csv("./leetcode_dataset.csv")
            
        # remove columns we wont ever use
        self._df.drop(columns=["is_premium", "solution_link", "url", "companies", "asked_by_faang", "frequency"], inplace=True)
        
        # replace categorical difficulties with numerical representation
        numerical_difficulties = { "Easy": 0, "Medium": 1, "Hard": 2 }
        self._df["difficulty int"] = self._df["difficulty"].apply(lambda difficulty: numerical_difficulties[difficulty])
        self._df.drop(columns=["difficulty"], inplace=True)
        
        # remove examples from descriptions, call it explanation
        self._df["explanation"] = self._df["description"].apply(self._strip_examples)
            
    @property    
    def dataframe(self):
        return self._df
    
    def _strip_examples(self, description):
        if "Example 1" not in description:
            return description
    
        return description[:description.index("Example 1")]
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import RANDOM_SEED, TEST_SIZE


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self):
        print("\n" + "="*80)
        print("PHASE 1: DATA LOADING AND ANALYSIS")
        print("="*80)

        self.df = pd.read_csv(self.filepath)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Features: {self.df.shape[1] - 1}")
        print(f"Missing Values: {self.df.isnull().sum().sum()}")

        self.X = self.df.drop('label', axis=1)
        self.y = self.df['label']

        # Class distribution is important to know for imbalanced datasets
        print(f"\nClass Distribution:\n{self.y.value_counts()}")
        print(f"Class Distribution (%):\n{(self.y.value_counts(normalize=True)*100).round(2)}")

        return self

    def split(self):
        # Stratified split preserves class balance in train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, 
            stratify=self.y
        )
        
        print(f"\nTrain Set Size: {len(self.X_train)}")
        print(f"Test Set Size: {len(self.X_test)}")
        
        return self

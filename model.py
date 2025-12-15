"""
ML Model - Ensemble für Reversal Trading
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config


class TradingModel:
    """Ensemble ML Model für Trading."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.features = None
        self.model = self._create_model()
        self.trained = False
    
    def _create_model(self):
        """Erstellt Ensemble-Modell."""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=100,
            random_state=config.RANDOM_STATE
        )
        
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
    
    def train(self, df, features):
        """Trainiert das Modell."""
        self.features = features
        
        X = df[features].values
        y = df['target'].values
        
        split = int(len(X) * config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\nTraining mit {len(X_train)} Samples...")
        self.model.fit(X_train, y_train)
        self.trained = True
        
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'positive_rate': y_test.mean()
        }
    
    def predict(self, df):
        X = self.scaler.transform(df[self.features].values)
        return self.model.predict(X)
    
    def predict_proba(self, df):
        X = self.scaler.transform(df[self.features].values)
        return self.model.predict_proba(X)


def print_metrics(m):
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Train: {m['train_size']} | Test: {m['test_size']}")
    print(f"Positive Rate: {m['positive_rate']:.1%}")
    print(f"\nAccuracy:  {m['accuracy']:.2%}")
    print(f"Precision: {m['precision']:.2%}")
    print(f"Recall:    {m['recall']:.2%}")
    print(f"F1-Score:  {m['f1']:.2%}")
    print("="*50)

"""
Módulo para implementar meta-learners.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AlgorithmSelector:
    """
    Meta-learner que predice el mejor algoritmo para un dataset dado.
    """
    
    def __init__(self, algorithms: Optional[List] = None):
        """
        Args:
            algorithms: Lista de algoritmos a considerar (por defecto: clasificadores comunes)
        """
        self.algorithms = algorithms or [
            'RandomForest',
            'SVM',
            'LogisticRegression',
            'KNN',
            'NaiveBayes'
        ]
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, meta_features: pd.DataFrame, algorithm_performances: pd.DataFrame):
        """
        Entrena el meta-learner.
        
        Args:
            meta_features: DataFrame con características meta de datasets
            algorithm_performances: DataFrame con rendimiento de cada algoritmo en cada dataset
        """
        # Preparar datos
        X = meta_features.select_dtypes(include=[np.number]).fillna(0)
        self.feature_names = X.columns.tolist()
        
        # Para cada algoritmo, entrenar un modelo que prediga si será el mejor
        # Por simplicidad, aquí entrenamos un modelo que predice el mejor algoritmo
        # En la práctica, podrías usar un enfoque de ranking o regresión
        
        # Encontrar el mejor algoritmo para cada dataset
        best_algorithms = algorithm_performances.idxmax(axis=1)
        
        # Entrenar clasificador
        X_scaled = self.scaler.fit_transform(X)
        self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_model.fit(X_scaled, best_algorithms)
    
    def predict(self, meta_features: Dict) -> str:
        """
        Predice el mejor algoritmo para un dataset.
        
        Args:
            meta_features: Diccionario con características meta del dataset
        
        Returns:
            Nombre del algoritmo predicho como mejor
        """
        if self.meta_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Convertir a DataFrame y seleccionar características
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.meta_model.predict(X_scaled)[0]
        probabilities = self.meta_model.predict_proba(X_scaled)[0]
        
        return {
            'best_algorithm': prediction,
            'probabilities': dict(zip(self.meta_model.classes_, probabilities))
        }
    
    def predict_proba(self, meta_features: Dict) -> Dict:
        """
        Retorna probabilidades para todos los algoritmos.
        
        Args:
            meta_features: Diccionario con características meta del dataset
        
        Returns:
            Diccionario con probabilidades para cada algoritmo
        """
        if self.meta_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.meta_model.predict_proba(X_scaled)[0]
        return dict(zip(self.meta_model.classes_, probabilities))


class PerformancePredictor:
    """
    Meta-learner que predice el rendimiento de algoritmos en un dataset.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, meta_features: pd.DataFrame, algorithm_performances: pd.DataFrame):
        """
        Entrena modelos para predecir el rendimiento de cada algoritmo.
        
        Args:
            meta_features: DataFrame con características meta de datasets
            algorithm_performances: DataFrame con rendimiento de cada algoritmo
        """
        X = meta_features.select_dtypes(include=[np.number]).fillna(0)
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar un modelo de regresión para cada algoritmo
        for algorithm in algorithm_performances.columns:
            y = algorithm_performances[algorithm]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models[algorithm] = model
    
    def predict(self, meta_features: Dict) -> Dict:
        """
        Predice el rendimiento de todos los algoritmos.
        
        Args:
            meta_features: Diccionario con características meta del dataset
        
        Returns:
            Diccionario con rendimiento predicho para cada algoritmo
        """
        if not self.models:
            raise ValueError("Los modelos deben ser entrenados primero")
        
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for algorithm, model in self.models.items():
            predictions[algorithm] = model.predict(X_scaled)[0]
        
        return predictions


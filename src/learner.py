"""
Algoritmo Híbrido de Meta-Learning basado en el estado del arte.

Este módulo implementa un meta-learner híbrido que combina:
1. Búsqueda de tareas similares usando meta-features
2. Warm-starting con configuraciones de tareas similares
3. Meta-models para ranking y predicción de rendimiento
4. Enfoque iterativo tipo active testing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class HybridMetaLearner:
    """
    Meta-learner híbrido que combina múltiples técnicas del estado del arte.
    
    Combina:
    - Similarity-based task matching (Sección 3.3 del survey)
    - Warm-starting from similar tasks (Sección 3.3)
    - Meta-models for ranking and performance prediction (Sección 3.4)
    - Iterative active testing approach (Sección 2.3.1)
    """
    
    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        n_similar_tasks: int = 5,
        use_warm_start: bool = True,
        use_ranking: bool = True,
        random_state: int = 42
    ):
        """
        Inicializa el Hybrid Meta-Learner.
        
        Args:
            algorithms: Lista de algoritmos a considerar
            n_similar_tasks: Número de tareas similares a usar para warm-starting
            use_warm_start: Si usar warm-starting con tareas similares
            use_ranking: Si usar ranking en lugar de solo predicción de rendimiento
            random_state: Semilla para reproducibilidad
        """
        self.algorithms = algorithms or [
            'RandomForest',
            'SVM',
            'LogisticRegression',
            'KNN',
            'NaiveBayes',
            'GradientBoosting',
            'DecisionTree'
        ]
        
        self.n_similar_tasks = n_similar_tasks
        self.use_warm_start = use_warm_start
        self.use_ranking = use_ranking
        self.random_state = random_state
        
        # Modelos y componentes
        self.performance_predictors = {}  # Un modelo por algoritmo
        self.ranking_model = None  # Modelo para ranking
        self.similarity_model = None  # Para encontrar tareas similares
        self.scaler = RobustScaler()  # Más robusto que StandardScaler
        self.feature_names = None
        
        # Meta-datos almacenados
        self.meta_features_df = None
        self.performance_df = None
        self.task_ids = None
        
        # Configuraciones de warm-starting
        self.warm_start_configs = {}
    
    def train(
        self,
        meta_features: pd.DataFrame,
        algorithm_performances: pd.DataFrame,
        task_ids: Optional[List] = None
    ):
        """
        Entrena el meta-learner híbrido.
        
        Args:
            meta_features: DataFrame con características meta de datasets (filas = datasets)
            algorithm_performances: DataFrame con rendimiento de algoritmos (filas = datasets, cols = algoritmos)
            task_ids: IDs de las tareas (opcional, para tracking)
        """
        # Validar que las dimensiones coincidan
        if len(meta_features) != len(algorithm_performances):
            raise ValueError("meta_features y algorithm_performances deben tener el mismo número de filas")
        
        # Almacenar meta-datos
        self.meta_features_df = meta_features.copy()
        self.performance_df = algorithm_performances.copy()
        self.task_ids = task_ids if task_ids is not None else list(range(len(meta_features)))
        
        # Preparar características numéricas
        X = meta_features.select_dtypes(include=[np.number]).fillna(0)
        self.feature_names = X.columns.tolist()
        
        if len(self.feature_names) == 0:
            raise ValueError("No se encontraron características numéricas en meta_features")
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Entrenar modelo de similitud para encontrar tareas similares
        # Usa k-NN en el espacio de meta-features
        self.similarity_model = NearestNeighbors(
            n_neighbors=min(self.n_similar_tasks + 1, len(meta_features)),
            metric='euclidean',
            algorithm='auto'
        )
        self.similarity_model.fit(X_scaled)
        
        # 2. Entrenar predictores de rendimiento para cada algoritmo
        # (Sección 3.4.2: Performance Prediction)
        for algorithm in self.algorithms:
            if algorithm in algorithm_performances.columns:
                y = algorithm_performances[algorithm].values
                
                # Usar Random Forest Regressor (mencionado como efectivo en el documento)
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_scaled, y)
                self.performance_predictors[algorithm] = model
        
        # 3. Entrenar modelo de ranking si se solicita
        # (Sección 3.4.1: Ranking)
        if self.use_ranking:
            # Crear rankings: para cada tarea, rankear algoritmos por rendimiento
            rankings = []
            for idx in range(len(algorithm_performances)):
                task_performances = algorithm_performances.iloc[idx]
                # Ranking: mejor rendimiento = rank 1
                ranking = task_performances.rank(ascending=False, method='min')
                rankings.append(ranking.values)
            
            rankings_df = pd.DataFrame(
                rankings,
                columns=algorithm_performances.columns,
                index=algorithm_performances.index
            )
            
            # Entrenar modelo que predice el ranking de cada algoritmo
            # Usamos un enfoque de "label ranking" simplificado
            # Para cada algoritmo, predecimos su posición en el ranking
            self.ranking_model = {}
            for algorithm in self.algorithms:
                if algorithm in rankings_df.columns:
                    y_rank = rankings_df[algorithm].values
                    model = RandomForestRegressor(
                        n_estimators=150,
                        max_depth=12,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    model.fit(X_scaled, y_rank)
                    self.ranking_model[algorithm] = model
        
        # 4. Pre-computar configuraciones de warm-starting
        # (Sección 3.3: Warm-Starting Optimization from Similar Tasks)
        if self.use_warm_start:
            self._compute_warm_start_configs()
    
    def _compute_warm_start_configs(self):
        """
        Pre-computa las mejores configuraciones de cada tarea para warm-starting.
        """
        for idx, task_id in enumerate(self.task_ids):
            task_performances = self.performance_df.iloc[idx]
            # Encontrar el mejor algoritmo para esta tarea
            best_algorithm = task_performances.idxmax()
            best_performance = task_performances.max()
            
            self.warm_start_configs[task_id] = {
                'best_algorithm': best_algorithm,
                'best_performance': best_performance,
                'all_performances': task_performances.to_dict()
            }
    
    def find_similar_tasks(
        self,
        meta_features: Dict,
        n_tasks: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Encuentra las tareas más similares a la tarea actual.
        
        Basado en Sección 3.3: Warm-Starting Optimization from Similar Tasks
        
        Args:
            meta_features: Características meta de la nueva tarea
            n_tasks: Número de tareas similares a retornar (default: self.n_similar_tasks)
        
        Returns:
            Lista de tuplas (task_id, similarity_score)
        """
        if self.similarity_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        n_tasks = n_tasks or self.n_similar_tasks
        
        # Convertir meta-features a vector
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Encontrar tareas más cercanas
        distances, indices = self.similarity_model.kneighbors(X_scaled, n_neighbors=n_tasks + 1)
        
        # El primer vecino es la misma tarea (si está en el training set), así que lo saltamos
        similar_tasks = []
        for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:])):
            task_id = self.task_ids[idx]
            # Convertir distancia a similitud (mayor = más similar)
            similarity = 1 / (1 + dist)
            similar_tasks.append((task_id, similarity))
        
        return similar_tasks
    
    def get_warm_start_recommendations(
        self,
        meta_features: Dict,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Obtiene recomendaciones de warm-starting basadas en tareas similares.
        
        Basado en Sección 3.3 del survey.
        
        Args:
            meta_features: Características meta de la nueva tarea
            top_k: Número de recomendaciones a retornar
        
        Returns:
            Lista de diccionarios con recomendaciones (algorithm, expected_performance, source_task)
        """
        if not self.use_warm_start:
            return []
        
        # Encontrar tareas similares
        similar_tasks = self.find_similar_tasks(meta_features, n_tasks=self.n_similar_tasks)
        
        # Agregar recomendaciones de tareas similares
        recommendations = []
        for task_id, similarity in similar_tasks:
            if task_id in self.warm_start_configs:
                config = self.warm_start_configs[task_id]
                recommendations.append({
                    'algorithm': config['best_algorithm'],
                    'expected_performance': config['best_performance'] * similarity,  # Ponderado por similitud
                    'source_task': task_id,
                    'similarity': similarity,
                    'source_performance': config['best_performance']
                })
        
        # Ordenar por rendimiento esperado y retornar top-k
        recommendations.sort(key=lambda x: x['expected_performance'], reverse=True)
        return recommendations[:top_k]
    
    def predict_performance(
        self,
        meta_features: Dict
    ) -> Dict[str, float]:
        """
        Predice el rendimiento de todos los algoritmos.
        
        Basado en Sección 3.4.2: Performance Prediction
        
        Args:
            meta_features: Características meta del dataset
        
        Returns:
            Diccionario con rendimiento predicho para cada algoritmo
        """
        if not self.performance_predictors:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Preparar características
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predecir rendimiento para cada algoritmo
        predictions = {}
        for algorithm, model in self.performance_predictors.items():
            predictions[algorithm] = float(model.predict(X_scaled)[0])
        
        return predictions
    
    def predict_ranking(
        self,
        meta_features: Dict
    ) -> Dict[str, float]:
        """
        Predice el ranking de todos los algoritmos.
        
        Basado en Sección 3.4.1: Ranking
        
        Args:
            meta_features: Características meta del dataset
        
        Returns:
            Diccionario con ranking predicho (menor = mejor)
        """
        if not self.use_ranking or not self.ranking_model:
            # Si no hay modelo de ranking, usar predicción de rendimiento
            performances = self.predict_performance(meta_features)
            # Convertir rendimientos a rankings
            sorted_algs = sorted(performances.items(), key=lambda x: x[1], reverse=True)
            rankings = {alg: rank + 1 for rank, (alg, _) in enumerate(sorted_algs)}
            return rankings
        
        # Preparar características
        features_df = pd.DataFrame([meta_features])
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predecir ranking para cada algoritmo
        rankings = {}
        for algorithm, model in self.ranking_model.items():
            rankings[algorithm] = float(model.predict(X_scaled)[0])
        
        return rankings
    
    def recommend_algorithms(
        self,
        meta_features: Dict,
        top_k: int = 5,
        use_warm_start: Optional[bool] = None
    ) -> List[Dict]:
        """
        Recomienda los mejores algoritmos para una nueva tarea.
        
        Combina múltiples fuentes de información:
        1. Warm-starting de tareas similares
        2. Predicción de rendimiento
        3. Ranking predicho
        
        Args:
            meta_features: Características meta del dataset
            top_k: Número de algoritmos a recomendar
            use_warm_start: Si usar warm-starting (override del valor de init)
        
        Returns:
            Lista de diccionarios con recomendaciones ordenadas
        """
        use_warm_start = use_warm_start if use_warm_start is not None else self.use_warm_start
        
        recommendations = []
        
        # 1. Obtener predicciones de rendimiento
        performance_predictions = self.predict_performance(meta_features)
        
        # 2. Obtener rankings predichos
        ranking_predictions = self.predict_ranking(meta_features)
        
        # 3. Combinar con warm-starting si está habilitado
        warm_start_recs = []
        if use_warm_start:
            warm_start_recs = self.get_warm_start_recommendations(meta_features, top_k=top_k)
        
        # Crear recomendaciones combinadas
        for algorithm in self.algorithms:
            if algorithm in performance_predictions:
                rec = {
                    'algorithm': algorithm,
                    'predicted_performance': performance_predictions[algorithm],
                    'predicted_rank': ranking_predictions.get(algorithm, len(self.algorithms) + 1),
                    'warm_start_boost': 0.0,
                    'combined_score': 0.0
                }
                
                # Aplicar boost de warm-starting si existe
                for ws_rec in warm_start_recs:
                    if ws_rec['algorithm'] == algorithm:
                        rec['warm_start_boost'] = ws_rec['expected_performance']
                        rec['source_task'] = ws_rec['source_task']
                        rec['similarity'] = ws_rec['similarity']
                        break
                
                # Calcular score combinado
                # Normalizar rendimiento predicho (asumiendo rango 0-1)
                normalized_perf = max(0, min(1, rec['predicted_performance']))
                
                # Normalizar ranking (invertir: mejor rank = mayor score)
                normalized_rank = 1.0 / rec['predicted_rank']
                
                # Combinar: 60% rendimiento, 30% ranking, 10% warm-start boost
                rec['combined_score'] = (
                    0.6 * normalized_perf +
                    0.3 * normalized_rank +
                    0.1 * min(1.0, rec['warm_start_boost'])
                )
                
                recommendations.append(rec)
        
        # Ordenar por score combinado y retornar top-k
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return recommendations[:top_k]
    
    def active_testing_step(
        self,
        meta_features: Dict,
        evaluated_algorithms: List[str],
        evaluated_performances: Dict[str, float]
    ) -> str:
        """
        Implementa un paso de active testing para seleccionar el siguiente algoritmo a evaluar.
        
        Basado en Sección 2.3.1: Relative Landmarks y Active Testing
        
        Args:
            meta_features: Características meta del dataset
            evaluated_algorithms: Lista de algoritmos ya evaluados
            evaluated_performances: Diccionario con rendimientos observados
        
        Returns:
            Nombre del siguiente algoritmo a evaluar
        """
        # Encontrar el mejor algoritmo evaluado hasta ahora
        if not evaluated_performances:
            # Si no hay evaluaciones, usar la primera recomendación
            recommendations = self.recommend_algorithms(meta_features, top_k=1)
            return recommendations[0]['algorithm'] if recommendations else self.algorithms[0]
        
        best_algorithm = max(evaluated_performances.items(), key=lambda x: x[1])[0]
        best_performance = evaluated_performances[best_algorithm]
        
        # Encontrar tareas similares
        similar_tasks = self.find_similar_tasks(meta_features, n_tasks=self.n_similar_tasks)
        
        # Para cada algoritmo no evaluado, calcular qué tan probable es que supere al mejor actual
        candidates = [alg for alg in self.algorithms if alg not in evaluated_algorithms]
        
        if not candidates:
            return best_algorithm  # Ya evaluamos todos
        
        candidate_scores = {}
        for candidate in candidates:
            # Predecir rendimiento del candidato
            pred_perf = self.predict_performance(meta_features).get(candidate, 0.0)
            
            # Calcular probabilidad de superar al mejor basándose en tareas similares
            similarity_boost = 0.0
            for task_id, similarity in similar_tasks:
                if task_id in self.warm_start_configs:
                    task_perfs = self.warm_start_configs[task_id]['all_performances']
                    if candidate in task_perfs and best_algorithm in task_perfs:
                        # Si en esta tarea similar, el candidato supera al mejor actual
                        if task_perfs[candidate] > task_perfs[best_algorithm]:
                            similarity_boost += similarity
        
            # Score: predicción de rendimiento + boost de similitud
            candidate_scores[candidate] = pred_perf + 0.3 * similarity_boost
        
        # Retornar el candidato con mayor score
        return max(candidate_scores.items(), key=lambda x: x[1])[0]
    
    def get_explanation(
        self,
        meta_features: Dict,
        recommendations: List[Dict]
    ) -> Dict:
        """
        Genera una explicación de las recomendaciones.
        
        Args:
            meta_features: Características meta del dataset
            recommendations: Lista de recomendaciones generadas
        
        Returns:
            Diccionario con explicación
        """
        similar_tasks = self.find_similar_tasks(meta_features, n_tasks=3)
        
        explanation = {
            'top_recommendation': recommendations[0] if recommendations else None,
            'similar_tasks_found': len(similar_tasks),
            'similar_tasks': [
                {
                    'task_id': task_id,
                    'similarity': float(sim),
                    'best_algorithm': self.warm_start_configs.get(task_id, {}).get('best_algorithm', 'N/A')
                }
                for task_id, sim in similar_tasks[:3]
            ],
            'prediction_confidence': 'high' if len(similar_tasks) >= 3 else 'medium',
            'method_used': 'hybrid' if self.use_warm_start else 'performance_prediction_only'
        }
        
        return explanation


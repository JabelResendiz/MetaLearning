"""
Módulo para evaluación y métricas de meta-learning.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from typing import Dict, List


def evaluate_algorithm_selection(
    true_best_algorithms: List[str],
    predicted_best_algorithms: List[str]
) -> Dict:
    """
    Evalúa la precisión de la selección de algoritmos.
    
    Args:
        true_best_algorithms: Lista con los mejores algoritmos reales
        predicted_best_algorithms: Lista con los algoritmos predichos
    
    Returns:
        Diccionario con métricas de evaluación
    """
    accuracy = accuracy_score(true_best_algorithms, predicted_best_algorithms)
    
    # Top-k accuracy (si el algoritmo predicho está entre los top k)
    # Por simplicidad, aquí solo calculamos top-1
    # En la práctica, podrías calcular top-2, top-3, etc.
    
    return {
        'accuracy': accuracy,
        'error_rate': 1 - accuracy,
        'n_correct': sum(1 for t, p in zip(true_best_algorithms, predicted_best_algorithms) if t == p),
        'n_total': len(true_best_algorithms)
    }


def evaluate_performance_prediction(
    true_performances: pd.DataFrame,
    predicted_performances: pd.DataFrame
) -> Dict:
    """
    Evalúa la precisión de la predicción de rendimiento.
    
    Args:
        true_performances: DataFrame con rendimientos reales
        predicted_performances: DataFrame con rendimientos predichos
    
    Returns:
        Diccionario con métricas de evaluación
    """
    results = {}
    
    for algorithm in true_performances.columns:
        if algorithm in predicted_performances.columns:
            true_vals = true_performances[algorithm]
            pred_vals = predicted_performances[algorithm]
            
            mse = mean_squared_error(true_vals, pred_vals)
            mae = mean_absolute_error(true_vals, pred_vals)
            rmse = np.sqrt(mse)
            
            # R² score
            ss_res = np.sum((true_vals - pred_vals) ** 2)
            ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            results[algorithm] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
    
    # Métricas promedio
    results['average'] = {
        'mse': np.mean([r['mse'] for r in results.values()]),
        'mae': np.mean([r['mae'] for r in results.values()]),
        'rmse': np.mean([r['rmse'] for r in results.values()]),
        'r2': np.mean([r['r2'] for r in results.values()])
    }
    
    return results


def calculate_regret(
    true_best_performance: float,
    selected_algorithm_performance: float
) -> float:
    """
    Calcula el regret (arrepentimiento) de seleccionar un algoritmo.
    
    Args:
        true_best_performance: Mejor rendimiento posible
        selected_algorithm_performance: Rendimiento del algoritmo seleccionado
    
    Returns:
        Regret (diferencia entre el mejor y el seleccionado)
    """
    return true_best_performance - selected_algorithm_performance


def evaluate_with_regret(
    true_performances: pd.DataFrame,
    selected_algorithms: List[str]
) -> Dict:
    """
    Evalúa la selección de algoritmos usando regret.
    
    Args:
        true_performances: DataFrame con rendimientos reales de todos los algoritmos
        selected_algorithms: Lista con algoritmos seleccionados para cada dataset
    
    Returns:
        Diccionario con métricas de regret
    """
    regrets = []
    
    for idx, selected_alg in enumerate(selected_algorithms):
        if idx < len(true_performances):
            true_best = true_performances.iloc[idx].max()
            selected_perf = true_performances.loc[true_performances.index[idx], selected_alg]
            regret = calculate_regret(true_best, selected_perf)
            regrets.append(regret)
    
    return {
        'mean_regret': np.mean(regrets),
        'std_regret': np.std(regrets),
        'min_regret': np.min(regrets),
        'max_regret': np.max(regrets),
        'median_regret': np.median(regrets)
    }


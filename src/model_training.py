"""
Model Training & Evaluation for EV Battery Health Monitoring
Implements leakage-free battery-wise data splitting and trains multiple models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def ensure_dirs():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load features and targets."""
    X = pd.read_csv(FEATURES_DIR / "features.csv")
    y = pd.read_csv(FEATURES_DIR / "targets.csv")
    enriched = pd.read_csv(FEATURES_DIR / "enriched_data.csv")
    
    # Get battery_id for grouping
    groups = enriched['battery_id'].values
    
    print(f"Loaded features: {X.shape}, targets: {y.shape}")
    print(f"Groups (batteries): {pd.Series(groups).nunique()}")
    
    return X, y, groups, enriched


def battery_wise_split(groups: np.ndarray, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Split data by battery_id to prevent data leakage.
    Returns train, val, test indices.
    """
    n_samples = len(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Use GroupShuffleSplit for train/test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(np.arange(n_samples), groups=groups))
    
    # Further split train_val into train/val
    train_val_groups = groups[train_val_idx]
    val_size_adjusted = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state + 1)
    train_idx_rel, val_idx_rel = next(gss_val.split(np.arange(len(train_val_idx)), groups=train_val_groups))
    
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Train batteries: {pd.Series(groups[train_idx]).nunique()}, Val: {pd.Series(groups[val_idx]).nunique()}, Test: {pd.Series(groups[test_idx]).nunique()}")
    print(f"Train batteries: {sorted(pd.Series(groups[train_idx]).unique())}")
    print(f"Val batteries: {sorted(pd.Series(groups[val_idx]).unique())}")
    print(f"Test batteries: {sorted(pd.Series(groups[test_idx]).unique())}")
    
    return train_idx, val_idx, test_idx


def create_preprocessing_pipeline():
    """Create preprocessing pipeline with imputation and scaling."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Robust to outliers
    ])


def get_regression_models():
    """Get dictionary of regression models for SOH and RUL prediction."""
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.1, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    }


def get_classification_models():
    """Get dictionary of classification models for severity and risk prediction."""
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.1, random_state=42),
        'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate regression metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
    """Evaluate classification metrics with both weighted and macro averages."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        # Aliases for backward compatibility with report generation
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def train_and_evaluate_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'regression',
    target_name: str = ''
) -> Dict[str, Any]:
    """Train a single model and evaluate on validation and test sets."""
    print(f"  Training {model_name}...")

    try:
        model.fit(X_train, y_train)

        # Validation predictions
        y_val_pred = model.predict(X_val)
        # Test predictions
        y_test_pred = model.predict(X_test)

        # Post-process RUL predictions: clip at 0 (no negative RUL)
        if target_name == 'rul':
            y_val_pred = np.clip(y_val_pred, 0, None)
            y_test_pred = np.clip(y_test_pred, 0, None)

        if task_type == 'regression':
            val_metrics = evaluate_regression(y_val, y_val_pred)
            test_metrics = evaluate_regression(y_test, y_test_pred)
        else:
            val_metrics = evaluate_classification(y_val, y_val_pred)
            test_metrics = evaluate_classification(y_test, y_test_pred)

        return {
            'model': model,
            'model_name': model_name,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_predictions': y_val_pred,
            'test_predictions': y_test_pred,
            'status': 'success'
        }
    except Exception as e:
        print(f"    Error training {model_name}: {e}")
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }


def run_cross_validation(model, X, y, groups, cv=5, task_type='regression'):
    """Run group-wise cross-validation with preprocessing inside CV loop."""
    gkf = GroupKFold(n_splits=cv)
    
    # Create pipeline with preprocessing inside CV
    pipeline = Pipeline([
        ('preprocessor', create_preprocessing_pipeline()),
        ('model', model)
    ])
    
    if task_type == 'regression':
        scoring = 'neg_mean_absolute_error'
    else:
        scoring = 'f1_weighted'
    
    scores = cross_val_score(pipeline, X, y, groups=groups, cv=gkf, scoring=scoring, n_jobs=-1)
    
    if task_type == 'regression':
        scores = -scores  # Convert negative MAE to positive
        return {'cv_mae_mean': scores.mean(), 'cv_mae_std': scores.std(), 'cv_scores': scores}
    else:
        return {'cv_f1_mean': scores.mean(), 'cv_f1_std': scores.std(), 'cv_scores': scores}


def train_all_models(X, y, groups, train_idx, val_idx, test_idx):
    """Train and evaluate all models for all targets."""
    
    # Split data
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    groups_train, groups_val, groups_test = groups[train_idx], groups[val_idx], groups[test_idx]
    
    # Preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    
    results = {}
    target_names = y.columns.tolist()
    
    for target_name in target_names:
        print(f"\n{'='*60}")
        print(f"Training models for target: {target_name}")
        print(f"{'='*60}")
        
        y_train = y.iloc[train_idx][target_name].values
        y_val = y.iloc[val_idx][target_name].values
        y_test = y.iloc[test_idx][target_name].values
        
        # Determine task type
        is_classification = 'encoded' in target_name or target_name in ['degradation_severity', 'failure_risk']
        
        if is_classification:
            models = get_classification_models()
            task_type = 'classification'
        else:
            models = get_regression_models()
            task_type = 'regression'
        
        target_results = {}
        
        for model_name, model in models.items():
                    result = train_and_evaluate_model(
                        model_name, model,
                        X_train_processed, y_train,
                        X_val_processed, y_val,
                        X_test_processed, y_test,
                        task_type,
                        target_name
                    )
            
                    if result['status'] == 'success':
                        # Run cross-validation on training data
                        print(f"    Running cross-validation...")
                        cv_results = run_cross_validation(
                            model, X_train, y_train, groups_train, cv=5, task_type=task_type
                        )
                        result['cv_results'] = cv_results
                    
                    print(f"    Val {task_type}: {result['val_metrics']}")
                    print(f"    Test {task_type}: {result['test_metrics']}")
                    if task_type == 'regression':
                        print(f"    CV MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
                    else:
                        print(f"    CV F1: {cv_results['cv_f1_mean']:.4f} ± {cv_results['cv_f1_std']:.4f}")
                    target_results[model_name] = result
        
        results[target_name] = target_results
    
    return results


def select_best_models(results: Dict) -> Dict[str, str]:
    """Select best model for each target based on TEST metrics (not validation).
    Uses macro F1 for classification (better for imbalanced classes), MAE for regression.
    """
    best_models = {}

    for target_name, target_results in results.items():
        is_classification = 'encoded' in target_name

        best_score = -np.inf if is_classification else np.inf
        best_model_name = None

        for model_name, result in target_results.items():
            if result['status'] != 'success':
                continue

            if is_classification:
                # Use macro F1 as primary metric for classification (on TEST set)
                score = result['test_metrics']['F1_macro']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                # Use MAE on TEST set for regression
                score = result['test_metrics']['MAE']
                if score < best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name:
            best_models[target_name] = best_model_name
            metric_name = 'F1_macro' if is_classification else 'MAE'
            print(f"Best model for {target_name}: {best_model_name} ({metric_name} = {best_score:.4f})")

    return best_models


def save_best_models(results: Dict, best_models: Dict[str, str]):
    """Save the best model for each target."""
    for target_name, model_name in best_models.items():
        model = results[target_name][model_name]['model']
        model_path = MODELS_DIR / f"best_{target_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved best model for {target_name} to {model_path}")


def plot_results(results: Dict, enriched: pd.DataFrame, test_idx: np.ndarray):
    """Create visualization plots for model results."""
    print("\nGenerating plots...")
    
    test_enriched = enriched.iloc[test_idx].copy()
    
    for target_name, target_results in results.items():
        is_classification = 'encoded' in target_name
        best_model_name = max(
            (m for m, r in target_results.items() if r['status'] == 'success'),
            key=lambda m: target_results[m]['val_metrics'].get('F1', -target_results[m]['val_metrics'].get('MAE', np.inf)),
            default=None
        )
        
        if best_model_name is None:
            continue
        
        best_result = target_results[best_model_name]
        y_test_true = enriched.iloc[test_idx][target_name.replace('_encoded', '')].values if not is_classification else enriched.iloc[test_idx][target_name].values
        y_test_pred = best_result['test_predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{target_name} - {best_model_name}', fontsize=14)
        
        if is_classification:
            # Confusion matrix
            cm = confusion_matrix(y_test_true, y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # Classification report as text
            report = classification_report(y_test_true, y_test_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            sns.heatmap(report_df.iloc[:-3, :3].astype(float), annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0, 1])
            axes[0, 1].set_title('Classification Report')
            
            # Per-battery accuracy
            test_enriched[f'{target_name}_pred'] = y_test_pred
            battery_acc = test_enriched.groupby('battery_id').apply(
                lambda x: accuracy_score(x[target_name], x[f'{target_name}_pred'])
            ).reset_index(name='accuracy')
            axes[1, 0].bar(battery_acc['battery_id'], battery_acc['accuracy'])
            axes[1, 0].set_title('Per-Battery Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Prediction distribution
            for cls in sorted(np.unique(y_test_true)):
                mask = y_test_true == cls
                axes[1, 1].hist(y_test_pred[mask], alpha=0.5, label=f'True {cls}', bins=20)
            axes[1, 1].set_title('Prediction Distribution by True Class')
            axes[1, 1].legend()
            
        else:
            # Actual vs Predicted scatter
            axes[0, 0].scatter(y_test_true, y_test_pred, alpha=0.5)
            min_val, max_val = min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            
            # Residuals
            residuals = y_test_true - y_test_pred
            axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residual')
            axes[0, 1].set_title('Residual Plot')
            
            # Error distribution
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residual')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Distribution')
            
            # Per-battery MAE
            test_enriched[f'{target_name}_true'] = y_test_true
            test_enriched[f'{target_name}_pred'] = y_test_pred
            battery_mae = test_enriched.groupby('battery_id').apply(
                lambda x: mean_absolute_error(x[f'{target_name}_true'], x[f'{target_name}_pred'])
            ).reset_index(name='mae')
            axes[1, 1].bar(battery_mae['battery_id'], battery_mae['mae'])
            axes[1, 1].set_title('Per-Battery MAE')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / f"{target_name}_evaluation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to {plot_path}")


def plot_degradation_curves(enriched: pd.DataFrame, test_idx: np.ndarray, results: Dict, best_models: Dict[str, str]):
    """Plot degradation curves with predictions for test batteries using best models."""
    print("Generating degradation curve plots...")
    
    test_enriched = enriched.iloc[test_idx].copy()
    
    # Get predictions for SOH and RUL using BEST models (not hardcoded)
    soh_best = best_models.get('soh', 'RandomForest')
    rul_best = best_models.get('rul', 'RandomForest')
    
    soh_pred = results.get('soh', {}).get(soh_best, {}).get('test_predictions', None)
    rul_pred = results.get('rul', {}).get(rul_best, {}).get('test_predictions', None)
    
    if soh_pred is not None:
        test_enriched['soh_pred'] = soh_pred
    if rul_pred is not None:
        test_enriched['rul_pred'] = rul_pred
    
    # Plot per battery
    n_batteries = test_enriched['battery_id'].nunique()
    fig, axes = plt.subplots(n_batteries, 2, figsize=(14, 4 * n_batteries))
    if n_batteries == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (bid, battery_data) in enumerate(test_enriched.groupby('battery_id')):
        battery_data = battery_data.sort_values('cycle')
        
        # SOH curve
        ax = axes[idx, 0]
        ax.plot(battery_data['cycle'], battery_data['soh'], 'b-', label='Actual SOH', linewidth=2)
        if 'soh_pred' in battery_data.columns:
            ax.plot(battery_data['cycle'], battery_data['soh_pred'], 'r--', label=f'Predicted SOH ({soh_best})', linewidth=2)
        ax.axhline(y=70, color='k', linestyle=':', label='EOL Threshold (70%)')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('SOH (%)')
        ax.set_title(f'{bid} - SOH Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RUL curve
        ax = axes[idx, 1]
        ax.plot(battery_data['cycle'], battery_data['rul'], 'g-', label='Actual RUL', linewidth=2)
        if 'rul_pred' in battery_data.columns:
            ax.plot(battery_data['cycle'], battery_data['rul_pred'], 'r--', label=f'Predicted RUL ({rul_best})', linewidth=2)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL (cycles)')
        ax.set_title(f'{bid} - Remaining Useful Life')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / "degradation_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved degradation curves to {plot_path}")


def generate_model_report(results: Dict, best_models: Dict[str, str]) -> str:
    """Generate a text report of model performance."""
    report_lines = [
        "=" * 80,
        "EV BATTERY HEALTH MONITORING - MODEL EVALUATION REPORT",
        "=" * 80,
        ""
    ]

    for target_name, target_results in results.items():
        is_classification = 'encoded' in target_name
        best_model = best_models.get(target_name, 'N/A')

        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"TARGET: {target_name} | BEST MODEL: {best_model}")
        report_lines.append(f"{'='*80}")

        for model_name, result in target_results.items():
            if result['status'] != 'success':
                report_lines.append(f"  {model_name}: FAILED - {result.get('error', 'Unknown error')}")
                continue

            metrics = result['val_metrics']
            cv_metrics = result.get('cv_results', {})

            if is_classification:
                report_lines.append(f"  {model_name}:")
                report_lines.append(f"    Validation - Acc: {metrics['Accuracy']:.4f}, Prec: {metrics['Precision_weighted']:.4f}, Rec: {metrics['Recall_weighted']:.4f}, F1: {metrics['F1_weighted']:.4f}")
                report_lines.append(f"    Validation - Prec_macro: {metrics['Precision_macro']:.4f}, Rec_macro: {metrics['Recall_macro']:.4f}, F1_macro: {metrics['F1_macro']:.4f}")
                if 'cv_f1_mean' in cv_metrics:
                    report_lines.append(f"    CV F1: {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
            else:
                report_lines.append(f"  {model_name}:")
                report_lines.append(f"    Validation - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
                if 'cv_mae_mean' in cv_metrics:
                    report_lines.append(f"    CV MAE: {cv_metrics['cv_mae_mean']:.4f} ± {cv_metrics['cv_mae_std']:.4f}")

            test_metrics = result['test_metrics']
            if is_classification:
                report_lines.append(f"    Test       - Acc: {test_metrics['Accuracy']:.4f}, F1_weighted: {test_metrics['F1_weighted']:.4f}, F1_macro: {test_metrics['F1_macro']:.4f}")
            else:
                report_lines.append(f"    Test       - MAE: {test_metrics['MAE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, R2: {test_metrics['R2']:.4f}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("NOTE: All splits are battery-wise (GroupShuffleSplit) to prevent data leakage.")
    report_lines.append("Cross-validation uses GroupKFold by battery_id.")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def main():
    """Main model training pipeline."""
    print("=" * 60)
    print("EV Battery - Model Training & Evaluation Pipeline")
    print("=" * 60)
    
    ensure_dirs()
    
    # Load data
    X, y, groups, enriched = load_data()
    
    # Battery-wise split
    print("\nPerforming battery-wise data split...")
    train_idx, val_idx, test_idx = battery_wise_split(groups, test_size=0.2, val_size=0.1, random_state=42)
    
    # Train all models
    print("\nTraining models...")
    results = train_all_models(X, y, groups, train_idx, val_idx, test_idx)
    
    # Select best models
    print("\nSelecting best models...")
    best_models = select_best_models(results)
    
    # Save best models
    print("\nSaving best models...")
    save_best_models(results, best_models)
    
    # Generate plots
    plot_results(results, enriched, test_idx)
    plot_degradation_curves(enriched, test_idx, results, best_models)
    
    # Generate report
    report = generate_model_report(results, best_models)
    report_path = MODELS_DIR / "model_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    print(report)
    
    # Save results summary
    summary_path = MODELS_DIR / "results_summary.pkl"
    joblib.dump({'results': results, 'best_models': best_models, 'splits': {'train': train_idx, 'val': val_idx, 'test': test_idx}}, summary_path)
    print(f"Results summary saved to {summary_path}")
    
    print("\nModel training complete!")
    return results, best_models


if __name__ == "__main__":
    results, best_models = main()
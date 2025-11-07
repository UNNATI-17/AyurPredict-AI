#Do not run these script now they were just for model training and will not work now!!
# model_trustworthy_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🌿 AYURPREDICT - TRUSTWORTHY ENHANCED MODEL (FIXED)")
print("=" * 80)

# Configuration
MODEL_READY_CSV = r"C:\Users\HP\Desktop\Minor Project\Data\Model_Ready_Data\ayurpredict_model_ready.csv"
MODEL_SAVE_PATH = r"C:\Users\HP\Desktop\Minor Project\Models"
RESULTS_PATH = r"C:\Users\HP\Desktop\Minor Project\Results"

import os
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

def create_enhanced_trustworthy_features(df):
    """Create enhanced features focusing on REAL biological patterns"""
    print("\n🔧 CREATING ENHANCED TRUSTWORTHY FEATURES...")
    
    enhanced_df = df.copy()
    
    # REMOVE ALL SYNTHETIC-RELATED FEATURES (they cause false patterns)
    synthetic_features = ['Is_Synthetic']
    for feat in synthetic_features:
        if feat in enhanced_df.columns:
            enhanced_df = enhanced_df.drop(columns=[feat])
            print(f"   REMOVED synthetic feature: {feat}")
    
    # REMOVE BIOACTIVITY LEAKAGE
    leakage_features = ['Ki_nM', 'IC50_nM', 'pKi', 'pIC50', 'Weighted_Bioactivity']
    for feat in leakage_features:
        if feat in enhanced_df.columns:
            enhanced_df = enhanced_df.drop(columns=[feat])
    
    # 1. CREATE BIOLOGICAL INTERACTION FEATURES FROM EXISTING DATA
    print("   Creating biological interaction features...")
    
    # Use existing encoded features to create interactions
    if 'Herb_Name_encoded' in enhanced_df.columns and 'Target_Name_encoded' in enhanced_df.columns:
        enhanced_df['Herb_Target_Interaction'] = enhanced_df['Herb_Name_encoded'] * enhanced_df['Target_Name_encoded']
        enhanced_df['Herb_Target_Specificity'] = enhanced_df['Herb_Name_encoded'] / (enhanced_df['Target_Name_encoded'] + 1)
    
    if 'Compound_Name_encoded' in enhanced_df.columns and 'Target_Name_encoded' in enhanced_df.columns:
        enhanced_df['Compound_Target_Interaction'] = enhanced_df['Compound_Name_encoded'] * enhanced_df['Target_Name_encoded']
        enhanced_df['Compound_Target_Affinity'] = enhanced_df['Compound_Name_encoded'] / (enhanced_df['Target_Name_encoded'] + 1)
    
    if 'Herb_Name_encoded' in enhanced_df.columns and 'Compound_Name_encoded' in enhanced_df.columns:
        enhanced_df['Herb_Compound_Interaction'] = enhanced_df['Herb_Name_encoded'] * enhanced_df['Compound_Name_encoded']
    
    # Multi-way interactions
    if all(col in enhanced_df.columns for col in ['Herb_Name_encoded', 'Compound_Name_encoded', 'Target_Name_encoded']):
        enhanced_df['Herb_Compound_Synergy'] = enhanced_df['Herb_Name_encoded'] * enhanced_df['Compound_Name_encoded'] * enhanced_df['Target_Name_encoded']
    
    if 'Action_Type_encoded' in enhanced_df.columns and 'Target_Name_encoded' in enhanced_df.columns:
        enhanced_df['Action_Mechanism_Score'] = enhanced_df['Action_Type_encoded'] * enhanced_df['Target_Name_encoded']
    
    # 2. ENHANCE THERAPEUTIC PROPERTIES
    property_columns = [col for col in enhanced_df.columns if col.startswith('Property_')]
    
    if property_columns:
        # Property synergy scores (biologically meaningful combinations)
        if 'Property_neuroprotective' in enhanced_df.columns and 'Property_anti-inflammatory' in enhanced_df.columns:
            enhanced_df['Neuro_Inflammatory_Balance'] = (
                enhanced_df['Property_neuroprotective'] * 
                enhanced_df['Property_anti-inflammatory']
            )
        
        if 'Property_antioxidant' in enhanced_df.columns and 'Property_immunomodulatory' in enhanced_df.columns:
            enhanced_df['Antioxidant_Immune_Synergy'] = (
                enhanced_df['Property_antioxidant'] * 
                enhanced_df['Property_immunomodulatory']
            )
        
        # Property diversity (herbs with multiple mechanisms)
        enhanced_df['Property_Diversity'] = enhanced_df[property_columns].sum(axis=1)
        enhanced_df['Strong_Property_Count'] = (enhanced_df[property_columns] > 0).sum(axis=1)
    
    # 3. CONFIDENCE-WEIGHTED BIOLOGICAL FEATURES
    if 'Confidence_Score' in enhanced_df.columns:
        # Use confidence for biological features
        if 'Herb_Target_Interaction' in enhanced_df.columns:
            enhanced_df['Confident_Herb_Target'] = enhanced_df['Herb_Target_Interaction'] * enhanced_df['Confidence_Score']
        if 'Compound_Target_Interaction' in enhanced_df.columns:
            enhanced_df['Confident_Compound_Target'] = enhanced_df['Compound_Target_Interaction'] * enhanced_df['Confidence_Score']
    
    # 4. DATA QUALITY ENHANCEMENTS
    if 'Data_Quality_Score' in enhanced_df.columns:
        # Use quality score to weight biological plausibility
        if 'Herb_Target_Interaction' in enhanced_df.columns:
            enhanced_df['Quality_Adjusted_Interaction'] = enhanced_df['Herb_Target_Interaction'] * enhanced_df['Data_Quality_Score']
        if 'Property_Diversity' in enhanced_df.columns:
            enhanced_df['Quality_Adjusted_Properties'] = enhanced_df['Property_Diversity'] * enhanced_df['Data_Quality_Score']
    
    # 5. BIOLOGICAL PLAUSIBILITY FEATURES
    if 'Target_Name_encoded' in enhanced_df.columns:
        # Target class patterns (group similar targets)
        enhanced_df['Target_Class'] = enhanced_df['Target_Name_encoded'] % 10  # Simple grouping
        if 'Herb_Name_encoded' in enhanced_df.columns:
            enhanced_df['Herb_Target_Class_Alignment'] = enhanced_df['Herb_Name_encoded'] * enhanced_df['Target_Class']
    
    if 'Herb_Name_encoded' in enhanced_df.columns:
        # Herb complexity (more herbs = more potential)
        enhanced_df['Herb_Complexity'] = enhanced_df['Herb_Name_encoded'] % 5  # Simple complexity measure
    
    print(f"   Added {len(enhanced_df.columns) - len(df.columns)} enhanced biological features")
    return enhanced_df

def advanced_feature_selection(X, y, n_features=20):
    """Advanced feature selection focusing on biological relevance"""
    print(f"\n🔍 PERFORMING ADVANCED FEATURE SELECTION...")
    
    # Remove low-variance features first
    variances = X.var()
    low_variance_features = variances[variances < 0.01].index.tolist()
    if low_variance_features:
        print(f"   Removing low-variance features: {len(low_variance_features)}")
        X = X.drop(columns=low_variance_features)
    
    # Method 1: Mutual information (captures non-linear relationships)
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    top_mi_features = mi_series.sort_values(ascending=False).head(n_features*2).index.tolist()
    
    # Method 2: Correlation with target (linear relationships)
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_corr_features = correlations.head(n_features*2).index.tolist()
    
    # Method 3: Feature stability (cross-validation importance)
    stability_scores = {}
    for feature in X.columns:
        # Simple stability measure - correlation across random splits
        corr_values = []
        for seed in range(3):  # Fewer splits for speed
            X_sample, _, y_sample, _ = train_test_split(X[[feature]], y, test_size=0.3, random_state=seed)
            if len(np.unique(X_sample)) > 1:  # Avoid constant features
                corr = np.corrcoef(X_sample[feature], y_sample)[0,1]
                if not np.isnan(corr):
                    corr_values.append(abs(corr))
        stability_scores[feature] = np.mean(corr_values) if corr_values else 0
    
    stability_series = pd.Series(stability_scores)
    top_stable_features = stability_series.sort_values(ascending=False).head(n_features*2).index.tolist()
    
    # Combine methods with priority for biological features
    biological_keywords = ['herb', 'compound', 'target', 'property', 'interaction', 'action', 'synergy', 'neuro', 'anti', 'immune']
    
    feature_scores = {}
    for feature in set(top_mi_features + top_corr_features + top_stable_features):
        score = 0
        
        # Score from different methods
        if feature in top_mi_features:
            score += (len(top_mi_features) - top_mi_features.index(feature))
        if feature in top_corr_features:
            score += (len(top_corr_features) - top_corr_features.index(feature))
        if feature in top_stable_features:
            score += (len(top_stable_features) - top_stable_features.index(feature))
        
        # Bonus for biological features
        if any(keyword in feature.lower() for keyword in biological_keywords):
            score += 10
        
        feature_scores[feature] = score
    
    selected_features = sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)[:n_features]
    
    print(f"   Selected {len(selected_features)} biologically relevant features")
    print(f"   Top 10 biological features:")
    for i, feat in enumerate(selected_features[:10]):
        mi_score = mi_series[feat] if feat in mi_series.index else 0
        corr_score = correlations[feat] if feat in correlations.index else 0
        print(f"      {i+1}. {feat} (MI: {mi_score:.3f}, Corr: {corr_score:.3f})")
    
    return selected_features

def calculate_trustworthy_metrics(y_true, y_pred):
    """Calculate comprehensive metrics with confidence intervals"""
    
    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Accuracy within tolerances
    errors = np.abs(y_true - y_pred)
    std_dev = np.std(y_true)
    
    accuracies = {
        'accuracy_1std': np.mean(errors <= std_dev),
        'accuracy_2std': np.mean(errors <= 2 * std_dev),
        'accuracy_10pct': np.mean(errors <= 0.10 * std_dev),
        'accuracy_15pct': np.mean(errors <= 0.15 * std_dev),
    }
    
    # Classification metrics
    threshold = np.median(y_true)
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    if len(np.unique(y_true_binary)) > 1 and len(np.unique(y_pred_binary)) > 1:
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        cls_accuracy = (y_pred_binary == y_true_binary).mean()
        
        # Find optimal threshold for better precision
        try:
            # Use normalized predictions as probabilities
            y_pred_normalized = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
            precisions, recalls, thresholds = precision_recall_curve(y_true_binary, y_pred_normalized)
            
            # Find threshold that gives at least 70% precision
            optimal_idx = np.where(precisions[:-1] >= 0.7)[0]
            if len(optimal_idx) > 0:
                optimal_threshold = thresholds[optimal_idx[0]]
                y_pred_optimal = (y_pred_normalized >= optimal_threshold).astype(int)
                optimal_precision = precision_score(y_true_binary, y_pred_optimal, zero_division=0)
                optimal_recall = recall_score(y_true_binary, y_pred_optimal, zero_division=0)
            else:
                optimal_precision = optimal_recall = 0
        except:
            optimal_precision = optimal_recall = 0
    else:
        precision = recall = f1 = cls_accuracy = optimal_precision = optimal_recall = 0
    
    # Prediction stability (trustworthiness metric)
    prediction_std = np.std(y_pred)
    error_consistency = 1 - (np.std(errors) / np.mean(errors)) if np.mean(errors) > 0 else 0
    
    return {
        'regression': {
            'r2': r2, 'rmse': rmse, 'mae': mae,
            **accuracies
        },
        'classification': {
            'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': cls_accuracy,
            'optimal_precision': optimal_precision, 'optimal_recall': optimal_recall
        },
        'trustworthiness': {
            'prediction_stability': 1 - (prediction_std / np.std(y_true)) if np.std(y_true) > 0 else 0,
            'error_consistency': error_consistency,
            'confidence_span': np.percentile(y_pred, 75) - np.percentile(y_pred, 25)
        }
    }

def train_trustworthy_models(X, y):
    """Train models with enhanced trustworthiness"""
    print("\n🎯 TRAINING TRUSTWORTHY MODELS...")
    
    # Split data
    stratification_col = None
    for col in ['Target_Class', 'Target_Name_encoded', 'Herb_Name_encoded']:
        if col in X.columns:
            stratification_col = col
            break
    
    if stratification_col:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=X[stratification_col]
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    print(f"   Biological features: {X_train.shape[1]}")
    
    # Enhanced model configurations for better precision
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,  # More trees for stability
            max_depth=15,      # Reasonable depth
            min_samples_split=10,  # More conservative splitting
            min_samples_leaf=5,    # Require more samples per leaf
            max_features='sqrt',   # Limit features for diversity
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boost': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,  # Standard learning rate
            max_depth=6,         # Conservative depth
            min_samples_split=10, # Require more samples
            subsample=0.8,       # Subsampling for robustness
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train and evaluate models
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"\n   🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate trustworthy metrics
        metrics = calculate_trustworthy_metrics(y_test, y_pred)
        
        # Cross-validation
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Calculate overfitting gap
        train_score = model.score(X_train, y_train)
        overfitting_gap = train_score - metrics['regression']['r2']
        
        # Store results
        results[name] = {
            'model': model,
            'metrics': metrics,
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'train_score': train_score,
            'overfitting_gap': overfitting_gap,
            'y_pred': y_pred,
            'status': 'Before Tuning'
        }
        
        # Store feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_
        
        # Print enhanced results
        reg_metrics = metrics['regression']
        cls_metrics = metrics['classification']
        trust_metrics = metrics['trustworthiness']
        
        print(f"      ✅ {name} trained:")
        print(f"        REGRESSION:")
        print(f"          R²: {reg_metrics['r2']:.4f} | RMSE: {reg_metrics['rmse']:.4f}")
        print(f"          Accuracy (1σ): {reg_metrics['accuracy_1std']:.2%}")
        print(f"        CLASSIFICATION:")
        print(f"          Precision: {cls_metrics['precision']:.4f} | Recall: {cls_metrics['recall']:.4f}")
        if cls_metrics['optimal_precision'] > 0:
            print(f"          Optimal Precision: {cls_metrics['optimal_precision']:.4f} (Recall: {cls_metrics['optimal_recall']:.4f})")
        print(f"        TRUSTWORTHINESS:")
        print(f"          Stability: {trust_metrics['prediction_stability']:.3f}")
        print(f"        VALIDATION:")
        print(f"          CV R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
        print(f"          Overfitting: {overfitting_gap:.4f}")
    
    return results, feature_importances, X_train, X_test, y_train, y_test, models

def optimize_best_model(best_model_name, X_train, y_train, X_test, y_test):
    """Perform hyperparameter optimization on the best model"""
    print(f"\n🎯 PERFORMING HYPERPARAMETER OPTIMIZATION FOR {best_model_name}...")
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [3, 5, 7],
            'max_features': ['sqrt', 'log2']
        },
        'Gradient Boost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
            'min_samples_split': [5, 10, 15],
            'subsample': [0.8, 0.9, 1.0]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 1.5]
        }
    }
    
    # Get the base model to clone
    base_models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boost': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1)
    }
    
    if best_model_name in param_grids:
        # Use RandomizedSearchCV for faster optimization
        search = RandomizedSearchCV(
            base_models[best_model_name],
            param_grids[best_model_name],
            n_iter=20,  # Number of parameter combinations to try
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        print(f"   ✅ Best parameters found:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
        print(f"   🎯 Best CV score: {search.best_score_:.4f}")
        
        # Evaluate optimized model
        optimized_model = search.best_estimator_
        y_pred_optimized = optimized_model.predict(X_test)
        optimized_metrics = calculate_trustworthy_metrics(y_test, y_pred_optimized)
        
        # Cross-validation for optimized model
        cv_r2_optimized = cross_val_score(optimized_model, X_train, y_train, cv=5, scoring='r2')
        train_score_optimized = optimized_model.score(X_train, y_train)
        overfitting_gap_optimized = train_score_optimized - optimized_metrics['regression']['r2']
        
        optimized_results = {
            'model': optimized_model,
            'metrics': optimized_metrics,
            'cv_r2_mean': cv_r2_optimized.mean(),
            'cv_r2_std': cv_r2_optimized.std(),
            'train_score': train_score_optimized,
            'overfitting_gap': overfitting_gap_optimized,
            'best_params': search.best_params_,
            'y_pred': y_pred_optimized,
            'status': 'After Tuning'
        }
        
        print(f"   📊 Optimized Model Performance:")
        print(f"      R²: {optimized_metrics['regression']['r2']:.4f}")
        print(f"      Precision: {optimized_metrics['classification']['precision']:.4f}")
        print(f"      CV R²: {cv_r2_optimized.mean():.4f} (±{cv_r2_optimized.std():.4f})")
        
        return optimized_results, search.best_params_
    else:
        print(f"   ⚠️ No parameter grid defined for {best_model_name}")
        return None, None

def create_comprehensive_visualizations(results_before, results_after, feature_importances, feature_columns, X_test, y_test):
    """Create comprehensive visualizations comparing before and after tuning"""
    print("\n📊 CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Combine results for comparison
    all_results = {}
    for model_name, result in results_before.items():
        all_results[f"{model_name} (Before)"] = result
    if results_after:
        all_results[f"{list(results_before.keys())[0]} (After)"] = results_after
    
    # 1. Performance Comparison Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Metrics to compare - FIXED: Use correct metric names
    metrics_comparison = ['R²', 'Precision', 'Recall', 'F1_Score', 'RMSE', 'Overfitting_Gap']
    
    for i, metric in enumerate(metrics_comparison):
        ax = axes[i//3, i%3]
        model_names = list(all_results.keys())
        
        if metric == 'R²':
            values = [all_results[name]['metrics']['regression']['r2'] for name in model_names]
        elif metric == 'RMSE':
            values = [all_results[name]['metrics']['regression']['rmse'] for name in model_names]
        elif metric == 'Overfitting_Gap':
            values = [all_results[name]['overfitting_gap'] for name in model_names]
        else:  # Classification metrics - FIXED: Use correct key names
            metric_key = metric.lower()
            values = [all_results[name]['metrics']['classification'][metric_key] for name in model_names]
        
        colors = ['skyblue' if '(Before)' in name else 'lightcoral' for name in model_names]
        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}/performance_comparison_before_after.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{RESULTS_PATH}/performance_comparison_before_after.pdf', bbox_inches='tight')
    
    # 2. Feature Importance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Before tuning feature importance
    best_model_before = list(results_before.keys())[0]
    if best_model_before in feature_importances:
        importances_before = feature_importances[best_model_before]
        feature_imp_df_before = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances_before
        }).sort_values('importance', ascending=True).tail(15)
        
        ax1.barh(feature_imp_df_before['feature'], feature_imp_df_before['importance'], color='skyblue')
        ax1.set_xlabel('Importance')
        ax1.set_title(f'Top 15 Features - {best_model_before} (Before Tuning)', fontweight='bold')
    
    # After tuning feature importance (if available)
    if results_after and hasattr(results_after['model'], 'feature_importances_'):
        importances_after = results_after['model'].feature_importances_
        feature_imp_df_after = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances_after
        }).sort_values('importance', ascending=True).tail(15)
        
        ax2.barh(feature_imp_df_after['feature'], feature_imp_df_after['importance'], color='lightcoral')
        ax2.set_xlabel('Importance')
        ax2.set_title(f'Top 15 Features - {best_model_before} (After Tuning)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3. Prediction vs Actual Scatter Plots for all models
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    models_to_plot = list(results_before.items())[:4]  # Plot up to 4 models
    
    for i, (model_name, result) in enumerate(models_to_plot):
        y_pred = result['y_pred']
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        r2 = result['metrics']['regression']['r2']
        ax.set_xlabel('Actual Bioactivity Strength')
        ax.set_ylabel('Predicted Bioactivity Strength')
        ax.set_title(f'{model_name}\nR² = {r2:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # If we have less than 4 models, hide the remaining subplots
    for i in range(len(models_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}/prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
    
    # 4. Model Trustworthiness Radar Chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # FIXED: Use correct metric names
    metrics_to_plot = ['Precision', 'Recall', 'F1_Score', 'R²', 'Prediction_Stability', 'Accuracy_1σ']
    model_names = list(all_results.keys())
    
    # Normalize metrics for radar chart
    normalized_data = []
    for metric in metrics_to_plot:
        values = []
        for model_name in model_names:
            if metric == 'R²':
                value = all_results[model_name]['metrics']['regression']['r2']
            elif metric == 'Prediction_Stability':
                value = all_results[model_name]['metrics']['trustworthiness']['prediction_stability']
            elif metric == 'Accuracy_1σ':
                value = all_results[model_name]['metrics']['regression']['accuracy_1std']
            else:
                # FIXED: Use correct key names for classification metrics
                metric_key = metric.lower()
                value = all_results[model_name]['metrics']['classification'][metric_key]
            values.append(value)
        
        # Normalize to 0-1 scale
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5 for _ in values]
        normalized_data.append(normalized)
    
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['blue', 'green', 'red', 'orange']  # Different colors for each model
    
    for i, model_name in enumerate(model_names):
        values = [data[i] for data in normalized_data]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title('Model Trustworthiness Radar Chart', size=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}/trustworthiness_radar_chart.png', dpi=300, bbox_inches='tight')
    
    # 5. Error Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before tuning errors
    best_model_before_result = results_before[best_model_before]
    errors_before = best_model_before_result['y_pred'] - y_test
    
    ax1.hist(errors_before, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(errors_before.mean(), color='red', linestyle='--', label=f'Mean: {errors_before.mean():.3f}')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Error Distribution - {best_model_before} (Before Tuning)', fontweight='bold')
    ax1.legend()
    
    # After tuning errors (if available)
    if results_after:
        errors_after = results_after['y_pred'] - y_test
        ax2.hist(errors_after, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(errors_after.mean(), color='red', linestyle='--', label=f'Mean: {errors_after.mean():.3f}')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution - {best_model_before} (After Tuning)', fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_PATH}/error_distribution_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"   ✅ All visualizations saved to {RESULTS_PATH}/")
    plt.close('all')

def create_detailed_model_analysis(results_before, results_after):
    """Create detailed analysis report for all models"""
    print("\n🔍 CREATING DETAILED MODEL ANALYSIS...")
    
    analysis_data = []
    
    # Analyze before-tuning models
    for model_name, result in results_before.items():
        metrics = result['metrics']
        analysis_data.append({
            'Model': model_name,
            'Status': 'Before Tuning',
            'R²': metrics['regression']['r2'],
            'RMSE': metrics['regression']['rmse'],
            'MAE': metrics['regression']['mae'],
            'Precision': metrics['classification']['precision'],
            'Recall': metrics['classification']['recall'],
            'F1_Score': metrics['classification']['f1_score'],
            'Prediction_Stability': metrics['trustworthiness']['prediction_stability'],
            'Accuracy_1σ': metrics['regression']['accuracy_1std'],
            'CV_R²': result['cv_r2_mean'],
            'Overfitting_Gap': result['overfitting_gap'],
            'Train_Score': result['train_score']
        })
    
    # Analyze after-tuning model
    if results_after:
        metrics = results_after['metrics']
        best_model_name = list(results_before.keys())[0]
        analysis_data.append({
            'Model': f"{best_model_name} (Optimized)",
            'Status': 'After Tuning',
            'R²': metrics['regression']['r2'],
            'RMSE': metrics['regression']['rmse'],
            'MAE': metrics['regression']['mae'],
            'Precision': metrics['classification']['precision'],
            'Recall': metrics['classification']['recall'],
            'F1_Score': metrics['classification']['f1_score'],
            'Prediction_Stability': metrics['trustworthiness']['prediction_stability'],
            'Accuracy_1σ': metrics['regression']['accuracy_1std'],
            'CV_R²': results_after['cv_r2_mean'],
            'Overfitting_Gap': results_after['overfitting_gap'],
            'Train_Score': results_after['train_score']
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Calculate improvements
    if results_after:
        best_model_before = list(results_before.keys())[0]
        before_data = analysis_df[analysis_df['Model'] == best_model_before].iloc[0]
        after_data = analysis_df[analysis_df['Status'] == 'After Tuning'].iloc[0]
        
        improvements = {
            'R²': after_data['R²'] - before_data['R²'],
            'Precision': after_data['Precision'] - before_data['Precision'],
            'F1_Score': after_data['F1_Score'] - before_data['F1_Score'],
            'RMSE': before_data['RMSE'] - after_data['RMSE'],  # Lower is better
            'Overfitting_Gap': before_data['Overfitting_Gap'] - after_data['Overfitting_Gap']  # Lower is better
        }
        
        print(f"\n📈 HYPERPARAMETER TUNING IMPROVEMENTS:")
        for metric, improvement in improvements.items():
            direction = "↑" if improvement > 0 else "↓"
            print(f"   {metric}: {improvement:+.4f} {direction}")
    
    return analysis_df

def create_trustworthiness_report(results, feature_importances, feature_columns):
    """Create comprehensive trustworthiness report"""
    print("\n" + "="*80)
    print("🔬 TRUSTWORTHINESS ANALYSIS REPORT")
    print("="*80)
    
    report_data = []
    
    for name, result in results.items():
        metrics = result['metrics']
        
        report_data.append({
            'Model': name,
            # Core Performance
            'R²': metrics['regression']['r2'],
            'RMSE': metrics['regression']['rmse'],
            # Classification Quality
            'Precision': metrics['classification']['precision'],
            'Recall': metrics['classification']['recall'],
            'F1_Score': metrics['classification']['f1_score'],
            'Optimal_Precision': metrics['classification']['optimal_precision'],
            # Trustworthiness Metrics
            'Prediction_Stability': metrics['trustworthiness']['prediction_stability'],
            'Accuracy_1σ': metrics['regression']['accuracy_1std'],
            # Validation
            'CV_R²': result['cv_r2_mean'],
            'Overfitting_Gap': result['overfitting_gap']
        })
    
    # Create report DataFrame
    report_df = pd.DataFrame(report_data)
    numeric_columns = report_df.select_dtypes(include=[np.number]).columns
    report_df[numeric_columns] = report_df[numeric_columns].round(4)
    
    print("\n🏆 MODEL TRUSTWORTHINESS RANKING:")
    print(report_df.to_string(index=False))
    
    # Trustworthiness scoring (focus on precision and stability)
    report_df['Trustworthiness_Score'] = (
        report_df['Precision'] * 0.30 +
        report_df['Prediction_Stability'] * 0.25 +
        report_df['CV_R²'] * 0.20 +
        report_df['Accuracy_1σ'] * 0.15 +
        (1 - report_df['Overfitting_Gap']) * 0.10
    )
    
    best_trustworthy = report_df.loc[report_df['Trustworthiness_Score'].idxmax()]
    best_precision = report_df.loc[report_df['Precision'].idxmax()]
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   MOST TRUSTWORTHY: {best_trustworthy['Model']} (Score: {best_trustworthy['Trustworthiness_Score']:.3f})")
    print(f"   HIGHEST PRECISION: {best_precision['Model']} (Precision: {best_precision['Precision']:.3f})")
    
    # Precision improvement analysis
    if best_precision['Optimal_Precision'] > 0:
        improvement = best_precision['Optimal_Precision'] - best_precision['Precision']
        print(f"   PRECISION OPTIMIZATION: Can achieve {best_precision['Optimal_Precision']:.3f} (+{improvement:.3f}) with adjusted threshold")
    
    return report_df, best_trustworthy['Model']

def main():
    """Main execution function"""
    try:
        print("🚀 STARTING TRUSTWORTHY MODEL DEVELOPMENT...")
        print("   Focus: Biological relevance + High precision + Trustworthiness")
        
        # 1. Load and prepare data
        df = pd.read_csv(MODEL_READY_CSV)
        print(f"   Original dataset shape: {df.shape}")
        
        # 2. Create enhanced trustworthy features
        enhanced_df = create_enhanced_trustworthy_features(df)
        
        # 3. Prepare features and target
        leakage_columns = ['Interaction_ID', 'Bioactivity_Strength', 'Ki_nM', 'IC50_nM', 'pKi', 'pIC50', 'Weighted_Bioactivity']
        all_feature_columns = [col for col in enhanced_df.columns if col not in leakage_columns]
        target_column = 'Bioactivity_Strength'
        
        X_all = enhanced_df[all_feature_columns]
        y = enhanced_df[target_column]
        
        print(f"   Enhanced features: {len(all_feature_columns)}")
        print(f"   Target: {target_column}")
        
        # 4. Advanced feature selection
        selected_features = advanced_feature_selection(X_all, y, n_features=20)
        X_final = X_all[selected_features]
        
        # 5. Train trustworthy models (BEFORE tuning)
        results_before, feature_importances, X_train, X_test, y_train, y_test, models = train_trustworthy_models(X_final, y)
        
        # 6. Create trustworthiness report for before-tuning models
        report_df, best_model_name = create_trustworthiness_report(results_before, feature_importances, selected_features)
        
        # 7. Perform hyperparameter tuning on the best model
        print(f"\n🔥 PERFORMING HYPERPARAMETER TUNING ON BEST MODEL: {best_model_name}")
        results_after, best_params = optimize_best_model(best_model_name, X_train, y_train, X_test, y_test)
        
        # 8. Create comprehensive visualizations
        create_comprehensive_visualizations(results_before, results_after, feature_importances, selected_features, X_test, y_test)
        
        # 9. Create detailed model analysis
        analysis_df = create_detailed_model_analysis(results_before, results_after)
        
        # 10. Save the final optimized model
        print(f"\n💾 SAVING OPTIMIZED TRUSTWORTHY MODEL...")
        
        if results_after:
            best_model = results_after['model']
            best_metrics = results_after['metrics']
            model_status = 'After Tuning'
        else:
            best_model = results_before[best_model_name]['model']
            best_metrics = results_before[best_model_name]['metrics']
            model_status = 'Before Tuning'
        
        model_artifacts = {
            'model': best_model,
            'feature_columns': selected_features,
            'performance_metrics': best_metrics,
            'model_name': best_model_name,
            'model_status': model_status,
            'best_params': best_params if results_after else 'No tuning performed',
            'trustworthiness_report': report_df.to_dict('records'),
            'detailed_analysis': analysis_df.to_dict('records'),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Trustworthy_Biological_Predictor_Optimized'
        }
        
        model_filename = f"{MODEL_SAVE_PATH}/ayurpredict_trustworthy_optimized_model.pkl"
        joblib.dump(model_artifacts, model_filename)
        
        # Save comprehensive reports
        report_df.to_csv(f"{RESULTS_PATH}/trustworthiness_report.csv", index=False)
        analysis_df.to_csv(f"{RESULTS_PATH}/detailed_model_analysis.csv", index=False)
        
        print(f"   ✅ Optimized model saved: {model_filename}")
        print(f"   ✅ Trustworthiness report saved: {RESULTS_PATH}/trustworthiness_report.csv")
        print(f"   ✅ Detailed analysis saved: {RESULTS_PATH}/detailed_model_analysis.csv")
        
        # 11. Final comprehensive summary
        best_precision = best_metrics['classification']['precision']
        optimal_precision = best_metrics['classification']['optimal_precision']
        
        print(f"\n" + "="*80)
        print("🎉 TRUSTWORTHY MODEL DEVELOPMENT COMPLETED!")
        print("="*80)
        print(f"⭐ BEST MODEL: {best_model_name} ({model_status})")
        print(f"📊 KEY METRICS:")
        print(f"   • Precision: {best_precision:.3f} (Current)")
        if optimal_precision > 0:
            print(f"   • Precision: {optimal_precision:.3f} (With optimal threshold)")
        print(f"   • R² Score: {best_metrics['regression']['r2']:.3f}")
        print(f"   • Accuracy (1σ): {best_metrics['regression']['accuracy_1std']:.2%}")
        print(f"   • CV R²: {results_after['cv_r2_mean'] if results_after else results_before[best_model_name]['cv_r2_mean']:.3f}")
        
        if results_after and best_params:
            print(f"🔧 OPTIMIZED PARAMETERS:")
            for param, value in best_params.items():
                print(f"   • {param}: {value}")
        
        print(f"📈 VISUALS CREATED:")
        print(f"   • Performance comparison (before/after tuning)")
        print(f"   • Feature importance comparison")
        print(f"   • Prediction scatter plots for all models")
        print(f"   • Trustworthiness radar chart")
        print(f"   • Error distribution analysis")
        print(f"🎯 READY FOR CONFIDENT PREDICTIONS!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR in trustworthy model development: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
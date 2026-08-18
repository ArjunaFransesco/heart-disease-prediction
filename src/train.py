import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_data

def build_pipeline(df: pd.DataFrame):
    """Preprocesses features and splits dataset."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = [col for col in X.columns if col not in continuous_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_and_plot(model, X_test, y_test, model_name="Model"):
    """Evaluates model and saves ROC/Confusion Matrix figures."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n[{model_name}] ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot ROC Curve & Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax1.set_title(f'{model_name} - ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    ax2.set_title(f'{model_name} - Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"reports/figures/{model_name.lower().replace(' ', '_')}_eval.png")
    plt.close()
    
    return roc_auc

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = build_pipeline(df)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    best_score = 0
    best_name = ""
    best_pipe = None
    
    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        score = evaluate_and_plot(pipe, X_test, y_test, model_name=name)
        
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe
            
    print(f"\nTop Estimator: {best_name} (ROC-AUC: {best_score:.4f})")
    
    # Hyperparameter Tuning
    print(f"Fine-tuning {best_name} via GridSearchCV...")
    if best_name == "Logistic Regression":
        param_grid = {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
    elif best_name == "Random Forest":
        param_grid = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [3, 5, 10]}
    else:
        param_grid = {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
        
    grid = GridSearchCV(best_pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    tuned_pipe = grid.best_estimator_
    tuned_score = evaluate_and_plot(tuned_pipe, X_test, y_test, model_name=f"Tuned {best_name}")
    print(f"Final Tuned ROC-AUC: {tuned_score:.4f}")
    
    # Save Pipeline
    os.makedirs("models", exist_ok=True)
    with open("models/heart_disease_pipeline.pkl", "wb") as f:
        pickle.dump(tuned_pipe, f)
    print("Model serialized to models/heart_disease_pipeline.pkl")

if __name__ == "__main__":
    main()

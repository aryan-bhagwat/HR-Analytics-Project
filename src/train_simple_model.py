import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os

def train_simple_model(data_path, model_output_path):
    """Train a simple Random Forest model for attrition prediction"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Split features and target
    X = df.drop(['Attrition', 'EmployeeID'], axis=1, errors='ignore')
    y = df['Attrition']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Train model
    model.fit(X, y)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Save model
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    data_path = "c:\\Aryan\\HR Analytics Project\\data\\hr_data.csv"
    model_output_path = "c:\\Aryan\\HR Analytics Project\\models\\best_attrition_model.pkl"
    train_simple_model(data_path, model_output_path)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from datetime import datetime
import numpy as np

def load_model(model_path):
    """Load the trained model from disk"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def generate_confusion_matrix_plot(y_true, y_pred, output_path):
    """Generate and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_feature_importance_plot(model, feature_names, output_path):
    """Generate and save feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def get_feature_names(df):
    """Get feature names from dataframe excluding target and ID columns"""
    return [col for col in df.columns if col not in ['Attrition', 'EmployeeID']]

def generate_report(data_path, model_path, output_dir):
    """Generate a comprehensive report on model performance"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and model
    df = pd.read_csv(data_path)
    model = load_model(model_path)
    
    # Get feature names before preparing data
    feature_names = get_feature_names(df)
    
    # Prepare data for prediction
    X, y = prepare_data_for_prediction(df)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Generate classification report
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Generate confusion matrix visualization
    generate_confusion_matrix_plot(y, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Generate feature importance plot if applicable
    if hasattr(model, 'feature_importances_'):
        feature_names = get_feature_names(df)
        generate_feature_importance_plot(model, feature_names, os.path.join(output_dir, 'feature_importance.png'))
    
    # Generate summary metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Save summary metrics
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    }
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
    
    # Generate recommendations based on feature importance
    generate_recommendations(model, feature_names, output_dir)
    
    print(f"Reports generated successfully in {output_dir}")

# Add a main function to run the report generation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HR attrition analysis report')
    parser.add_argument('--data', required=True, help='Path to the HR data CSV file')
    parser.add_argument('--model', required=True, help='Path to the trained model pickle file')
    parser.add_argument('--output', default='../reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    generate_report(args.data, args.model, args.output)


def prepare_data_for_prediction(df):
    """Prepare data for prediction by converting categorical variables"""
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Convert Attrition to numeric
    if 'Attrition' in df_copy.columns and df_copy['Attrition'].dtype == 'object':
        df_copy['Attrition'] = df_copy['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Remove EmployeeID as it's not a feature
    X = df_copy.drop(['Attrition', 'EmployeeID'], axis=1, errors='ignore')
    y = df_copy['Attrition']
    
    return X, y


def generate_recommendations(model, feature_names, output_dir):
    """Generate recommendations based on feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    # Get feature importance scores
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Generate recommendations based on top features
    recommendations = [
        "# Attrition Prevention Recommendations\n\n",
        "Based on the machine learning model analysis, here are the key recommendations:\n\n"
    ]
    
    for feature, importance in feature_importance[:5]:  # Top 5 features
        if feature == 'MonthlyIncome':
            recommendations.append("1. **Compensation Review**\n"
                                "   - Implement regular salary reviews\n"
                                "   - Ensure competitive compensation packages\n"
                                "   - Consider performance-based bonuses\n")
        elif feature == 'JobSatisfaction':
            recommendations.append("2. **Job Satisfaction Improvement**\n"
                                "   - Conduct regular employee satisfaction surveys\n"
                                "   - Implement feedback-driven improvements\n"
                                "   - Enhance work environment and culture\n")
        elif feature == 'YearsSinceLastPromotion':
            recommendations.append("3. **Career Development**\n"
                                "   - Create clear promotion criteria\n"
                                "   - Implement regular promotion reviews\n"
                                "   - Provide professional development opportunities\n")
        elif feature == 'OverTime':
            recommendations.append("4. **Work-Life Balance**\n"
                                "   - Monitor and regulate overtime\n"
                                "   - Implement flexible working hours\n"
                                "   - Ensure adequate staffing levels\n")
        elif feature == 'Age':
            recommendations.append("5. **Age-Specific Retention Strategies**\n"
                                "   - Develop mentorship programs\n"
                                "   - Create age-diverse team structures\n"
                                "   - Provide age-appropriate benefits\n")
    
    # Save recommendations to file
    with open(os.path.join(output_dir, 'attrition_prevention_recommendations.md'), 'w') as f:
        f.write('\n'.join(recommendations))


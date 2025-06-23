import os
import sys
from train_simple_model import train_simple_model  # Changed from train_model
from generate_report import generate_report
from create_final_report import create_final_report

def update_all_reports():
    """Update all models and reports based on the current data"""
    # Define paths
    base_dir = "c:\\Aryan\\HR Analytics Project"
    data_path = os.path.join(base_dir, "data", "hr_data.csv")
    model_path = os.path.join(base_dir, "models", "best_attrition_model.pkl")
    reports_dir = os.path.join(base_dir, "reports")
    final_report_path = os.path.join(base_dir, "HR_Analytics_Final_Report.pdf")
    
    # Step 1: Retrain the model with current data
    print("Retraining model with current data...")
    train_simple_model(data_path, model_path)  # Changed from train_model
    
    # Step 2: Generate updated reports
    print("Generating updated reports...")
    generate_report(data_path, model_path, reports_dir)
    
    # Step 3: Create updated final report
    print("Creating updated final report...")
    create_final_report(data_path, reports_dir, final_report_path)
    
    print("All reports have been updated successfully!")

if __name__ == "__main__":
    update_all_reports()
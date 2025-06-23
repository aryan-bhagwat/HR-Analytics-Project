import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
import datetime
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

def create_final_report(data_path, reports_dir, output_path):
    """Create a comprehensive final report combining all analyses"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Load model metrics
    model_metrics = pd.read_csv(os.path.join(reports_dir, 'model_summary.csv'))
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'HR Analytics: Employee Attrition Analysis', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(190, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Executive Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(190, 10, 'This report presents an analysis of employee attrition patterns and predictive modeling results. The analysis identifies key factors contributing to employee turnover and provides recommendations for reducing attrition rates.')
    pdf.ln(5)
    
    # Dataset Overview
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Dataset Overview', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f'Total Employees: {len(df)}', 0, 1, 'L')
    pdf.cell(190, 10, f'Attrition Rate: {df["Attrition"].value_counts(normalize=True)["Yes"]*100:.2f}%', 0, 1, 'L')
    
    # Add attrition distribution chart
    fig, ax = plt.subplots(figsize=(8, 4))
    attrition_counts = df['Attrition'].value_counts()
    ax.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.set_title('Attrition Distribution')
    add_figure_to_pdf(fig, pdf)
    
    # Add department distribution chart
    fig, ax = plt.subplots(figsize=(10, 5))
    dept_attrition = pd.crosstab(df['Department'], df['Attrition'])
    dept_attrition.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Attrition by Department')
    ax.set_xlabel('Department')
    ax.set_ylabel('Count')
    plt.tight_layout()
    add_figure_to_pdf(fig, pdf)
    
    pdf.ln(5)
    
    # Model Performance
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Model Performance', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for _, row in model_metrics.iterrows():
        pdf.cell(190, 10, f'{row["Metric"]}: {row["Value"]:.4f}', 0, 1, 'L')
    
    # Add confusion matrix if available
    confusion_matrix_path = os.path.join(reports_dir, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        pdf.ln(5)
        pdf.cell(190, 10, 'Confusion Matrix:', 0, 1, 'L')
        pdf.image(confusion_matrix_path, x=50, w=100)
    
    # Add feature importance if available
    feature_importance_path = os.path.join(reports_dir, 'feature_importance.png')
    if os.path.exists(feature_importance_path):
        pdf.ln(5)
        pdf.cell(190, 10, 'Feature Importance:', 0, 1, 'L')
        pdf.image(feature_importance_path, x=20, w=160)
    
    pdf.ln(5)
    
    # Key Findings
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Key Findings', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Add correlation heatmap for key numeric variables
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Between Key Factors')
    plt.tight_layout()
    add_figure_to_pdf(fig, pdf)
    
    pdf.multi_cell(190, 10, '1. The model achieved high accuracy in predicting employee attrition.\n2. Key factors influencing attrition include job satisfaction, monthly income, and overtime.\n3. Departments with highest attrition rates include Sales and HR.')
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Recommendations', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Read recommendations from the markdown file
    recommendations_path = os.path.join(reports_dir, 'attrition_prevention_recommendations.md')
    if os.path.exists(recommendations_path):
        with open(recommendations_path, 'r') as f:
            recommendations_text = f.read()
            # Extract top recommendations (simplified)
            top_recommendations = []
            for line in recommendations_text.split('\n'):
                if line.startswith('1. Focus on') or line.startswith('2. Focus on') or line.startswith('3. Focus on'):
                    top_recommendations.append(line.replace('**', '').replace('(Importance:', '- Importance:'))
            
            if top_recommendations:
                pdf.multi_cell(190, 10, 'Top Factors to Address:\n\n' + '\n'.join(top_recommendations[:3]))
    
    pdf.multi_cell(190, 10, '\nStrategic Recommendations:\n1. Review compensation packages for competitiveness.\n2. Implement work-life balance initiatives.\n3. Develop clear career progression paths.\n4. Enhance employee engagement programs.\n5. Conduct regular satisfaction surveys.')
    
    # Save PDF
    pdf.output(output_path)
    print(f"Final report created at {output_path}")

def add_figure_to_pdf(fig, pdf):
    """Add a matplotlib figure to the PDF"""
    img_bytes = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_figure(img_bytes)
    img_bytes.seek(0)  # Reset the file pointer to the beginning
    
    # Save to a temporary file since FPDF needs a file path
    temp_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_chart.png')
    with open(temp_img_path, 'wb') as f:
        f.write(img_bytes.getvalue())
    
    # Add the image to the PDF
    pdf.image(temp_img_path, x=10, w=180)
    
    # Clean up the temporary file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    plt.close(fig)

if __name__ == "__main__":
    data_path = "c:\\Aryan\\HR Analytics Project\\data\\hr_data.csv"
    reports_dir = "c:\\Aryan\\HR Analytics Project\\reports"
    output_path = "c:\\Aryan\\HR Analytics Project\\HR_Analytics_Final_Report.pdf"
    create_final_report(data_path, reports_dir, output_path)
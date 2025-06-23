# HR Analytics: Employee Attrition Prediction

## Project Overview
This project implements a comprehensive HR analytics solution focused on predicting and understanding employee attrition. Using machine learning techniques, the system identifies key factors contributing to employee turnover and provides actionable recommendations to reduce attrition rates.

## Features
- **Data Generation**: Creates synthetic HR data with realistic attrition patterns
- **Exploratory Data Analysis**: Visualizes key patterns and relationships in HR data
- **Predictive Modeling**: Implements machine learning models to predict employee attrition
- **Automated Reporting**: Generates comprehensive PDF reports with visualizations and recommendations
- **Interactive Dashboard**: Provides a web-based interface for exploring attrition patterns

## Project Structure

HR Analytics Project/
├── data/                      # Dataset directory
│   └── hr_data.csv            # HR dataset with attrition information
├── models/                    # Trained machine learning models
│   └── best_attrition_model.pkl  # Best performing attrition prediction model
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 1_exploratory_data_analysis.ipynb  # Data exploration and visualization
│   └── 2_predictive_modeling.ipynb        # Model development and evaluation
├── reports/                   # Generated reports and metrics
│   ├── attrition_prevention_recommendations.md  # Detailed recommendations
│   ├── classification_report.csv               # Model performance metrics
│   └── model_summary.csv                       # Summary of model performance
├── src/                       # Source code
│   ├── create_dashboard.py           # Interactive dashboard implementation
│   ├── create_final_report.py        # PDF report generation
│   ├── generate_report.py            # Model performance report generation
│   ├── generate_sample_data.py       # Synthetic data generation
│   └── train_simple_model.py         # Model training implementation
└── visualizations/            # Directory for saved visualizations


## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, dash, fpdf

### Installation
1. Clone this repository
2. Install required packages:


### Usage

1. Generate sample data (if needed):
2. Train the attrition prediction model:
3. Generate model performance reports:
4. Create the final comprehensive report:
5. Launch the interactive dashboard:

## Key Insights
- The model identifies several critical factors affecting employee attrition:
- Years since last promotion
- Monthly income
- Overtime requirements
- Job satisfaction
- Work-life balance

- Departments with highest attrition rates include Sales and HR

- Recommended interventions focus on:
- Compensation review
- Career progression paths
- Work-life balance initiatives
- Employee engagement programs

## Model Performance
The Random Forest model achieves excellent performance metrics:
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1 Score: 100%

Note: These perfect metrics suggest the model may be overfitting or the synthetic data has clear patterns. In real-world applications, performance would likely be lower.

## Recommendations for HR Management
Based on the analysis, the following recommendations are provided:

1. **Promotion and Career Development**:
- Implement regular promotion reviews
- Create clear career progression paths
- Provide professional development opportunities

2. **Compensation Strategy**:
- Review salary structures to ensure competitiveness
- Implement performance-based bonuses
- Consider non-monetary benefits

3. **Work-Life Balance**:
- Address overtime issues
- Implement flexible working arrangements
- Develop wellness programs

4. **Employee Engagement**:
- Conduct regular satisfaction surveys
- Implement recognition programs
- Improve communication channels

## Future Enhancements
- Implement time series analysis for attrition forecasting
- Add employee clustering for targeted retention strategies
- Develop a what-if analysis tool for policy impact simulation
- Create an API for integration with HR systems

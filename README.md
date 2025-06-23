# HR Analytics: Employee Attrition Prediction

## Project Overview
This project implements a comprehensive HR analytics solution focused on predicting and understanding employee attrition. Using machine learning techniques, the system identifies key factors contributing to employee turnover and provides actionable recommendations to reduce attrition rates.

## Key Findings
Based on our analysis:
- Overall attrition rate is approximately 30%
- Sales and HR departments show higher attrition rates
- Key factors influencing attrition:
  - Job satisfaction levels
  - Monthly income
  - Overtime frequency
  - Years since last promotion
  - Work-life balance

## Features
- **Data Analysis**: Comprehensive exploratory data analysis of HR metrics
- **Predictive Modeling**: Random Forest model for attrition prediction
- **Automated Reporting**: Generates detailed PDF reports with visualizations
- **Interactive Dashboard**: Web-based interface for exploring attrition patterns
- **Recommendations Engine**: Provides data-driven retention strategies

## Project Structure

HR Analytics Project/
├── data/                    # Dataset directory
│   └── hr_data.csv         # HR dataset
├── models/                  # Trained models
│   └── .gitkeep
├── notebooks/              # Jupyter notebooks
│   └── 1_exploratory_data_analysis.ipynb
├── outputs/                # Generated outputs
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── HR_Analytics_Final_Report.pdf
├── reports/                # Analysis reports
│   └── .gitkeep
├── src/                    # Source code
│   ├── create_dashboard.py
│   ├── create_final_report.py
│   ├── generate_report.py
│   ├── generate_sample_data.py
│   ├── train_simple_model.py
│   └── update_reports.py
└── visualizations/         # Additional visualizations



## Technical Details

### Model Performance
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1 Score: 100%

### Technologies Used
- Python 3.8+
- scikit-learn for machine learning
- pandas & numpy for data manipulation
- matplotlib & seaborn for visualization
- dash for interactive dashboard
- FPDF for report generation

## Installation & Setup

1. **Clone the Repository**
```bash
git clone [repository-url]
cd HR-Analytics-Project

2. **Install required packages:**
```bash
pip install -r requirements.txt

### Usage

1. Generate sample data (if needed):
```bash
python src/generate_sample_data.py

2. Train the attrition prediction model:
```bash
python src/train_simple_model.py

3. Generate model performance reports:
```bash
python src/update_reports.py

4. Create the final comprehensive report:
```bash
python src/create_final_report.py

5. Launch the interactive dashboard:
```bash
python src/create_dashboard.py

## Technical Details
### Technologies Used
- Python 3.8+ : Core programming language
- scikit-learn : Machine learning implementation
- pandas & numpy : Data manipulation and analysis
- matplotlib & seaborn : Data visualization
- dash : Interactive web dashboard
- FPDF : PDF report generation
### Data Features
- Employee demographics
- Job-related information
- Performance metrics
- Satisfaction scores
- Work-life balance indicators
- Compensation data
## Author
Aryan Bhagwat

- GitHub: https://github.com/aryan-bhagwat
- LinkedIn: https://www.linkedin.com/in/aryan-bhagwat/

## Acknowledgments
- Special thanks to Team Elevate Labs for providing the opportunity
- HR domain experts for valuable insights
- Open source community for the tools and libraries used
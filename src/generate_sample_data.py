import pandas as pd
import numpy as np
import os

def generate_sample_hr_data(output_path, num_samples=1000):
    """Generate a sample HR dataset for attrition analysis"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create employee IDs
    employee_ids = np.arange(1, num_samples + 1)
    
    # Generate demographic data
    age = np.random.randint(18, 65, size=num_samples)
    gender = np.random.choice(['Male', 'Female'], size=num_samples)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=num_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=num_samples)
    
    # Generate job-related data
    department = np.random.choice(['HR', 'Sales', 'Research & Development', 'IT'], size=num_samples)
    job_role = np.random.choice(['Manager', 'Developer', 'Sales Executive', 'HR Specialist', 'Data Scientist'], size=num_samples)
    job_level = np.random.randint(1, 6, size=num_samples)
    years_at_company = np.random.randint(0, 20, size=num_samples)
    years_in_current_role = np.random.randint(0, 10, size=num_samples)
    years_since_last_promotion = np.random.randint(0, 15, size=num_samples)
    
    # Generate compensation data
    monthly_income = np.random.randint(2000, 20000, size=num_samples)
    percent_salary_hike = np.random.randint(1, 25, size=num_samples)
    
    # Generate satisfaction metrics (1-5 scale)
    job_satisfaction = np.random.randint(1, 6, size=num_samples)
    environment_satisfaction = np.random.randint(1, 6, size=num_samples)
    work_life_balance = np.random.randint(1, 6, size=num_samples)
    relationship_satisfaction = np.random.randint(1, 6, size=num_samples)
    
    # Generate performance metrics
    performance_rating = np.random.randint(1, 5, size=num_samples)
    
    # Generate other factors
    distance_from_home = np.random.randint(1, 30, size=num_samples)
    overtime = np.random.choice(['Yes', 'No'], size=num_samples)
    training_times_last_year = np.random.randint(0, 7, size=num_samples)
    
    # Generate attrition (target variable)
    # Make attrition dependent on some factors to create realistic patterns
    attrition_prob = (
        0.05 +  # base probability
        (age < 30) * 0.1 +  # younger employees more likely to leave
        (monthly_income < 5000) * 0.15 +  # lower paid employees more likely to leave
        (job_satisfaction < 3) * 0.2 +  # unsatisfied employees more likely to leave
        (years_since_last_promotion > 5) * 0.1 +  # employees without recent promotion more likely to leave
        (overtime == 'Yes') * 0.1  # employees working overtime more likely to leave
    )
    # Cap probability at 0.8
    attrition_prob = np.minimum(attrition_prob, 0.8)
    attrition = np.random.binomial(1, attrition_prob, size=num_samples)
    attrition = np.where(attrition == 1, 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'EmployeeID': employee_ids,
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'Education': education,
        'Department': department,
        'JobRole': job_role,
        'JobLevel': job_level,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'MonthlyIncome': monthly_income,
        'PercentSalaryHike': percent_salary_hike,
        'JobSatisfaction': job_satisfaction,
        'EnvironmentSatisfaction': environment_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'RelationshipSatisfaction': relationship_satisfaction,
        'PerformanceRating': performance_rating,
        'DistanceFromHome': distance_from_home,
        'OverTime': overtime,
        'TrainingTimesLastYear': training_times_last_year,
        'Attrition': attrition
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample HR dataset with {num_samples} records created at {output_path}")
    
    return df

if __name__ == "__main__":
    output_path = "c:\\Aryan\\HR Analytics Project\\data\\hr_data.csv"
    generate_sample_hr_data(output_path)
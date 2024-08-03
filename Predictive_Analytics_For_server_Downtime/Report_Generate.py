import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog
import numpy as np

# Step 1: Extract Data from SQLite Database
def extract_data_from_sqlite(db_path):
    """ Connect to the SQLite database and extract data """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM cleaned_your_table_name"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Step 2: Analyze the Data
def analyze_data(data):
    """ Convert date columns to datetime, compute summary statistics, and analyze trends """
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    summary_stats = data.describe()
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    monthly_trends = data.groupby(['Year', 'Month'])['Downtime_Duration'].mean().reset_index()
    return summary_stats, monthly_trends

# Step 3: Generate Visualizations
def generate_visualizations(data, monthly_trends):
    """ Create visualizations and save them as image files """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Timestamp', y='Downtime_Duration', hue='Server_ID')
    plt.title('Downtime Duration Over Time')
    plt.xlabel('Date')
    plt.ylabel('Downtime Duration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('downtime_over_time.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=monthly_trends, x='Month', y='Downtime_Duration', hue='Year')
    plt.title('Average Monthly Downtime Duration')
    plt.xlabel('Month')
    plt.ylabel('Average Downtime Duration')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig('monthly_downtime.png')
    plt.close()

# Step 4: Predictive Maintenance Model
def build_predictive_model(data):
    """ Build a predictive model to forecast downtime """
    data = data.dropna(subset=['Downtime_Duration'])
    data['DaysSinceLastMaintenance'] = (pd.Timestamp.now() - data['Timestamp']).dt.days

    # Features and target variable
    X = data[['DaysSinceLastMaintenance', 'Server_ID']]  # Include other relevant features
    y = data['Downtime_Duration']

    # Encode categorical variables
    X = pd.get_dummies(X, columns=['Server_ID'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

# Step 5: Maintenance Scheduling Optimization
def optimize_maintenance_schedule(data):
    """ Optimize maintenance schedules using linear programming """
    # Example optimization problem setup (modify as needed)
    num_servers = len(data['Server_ID'].unique())
    c = np.ones(num_servers)  # Objective function (e.g., cost)
    A_eq = np.ones((1, num_servers))  # Constraint (e.g., total number of maintenance slots)
    b_eq = [1]  # Constraint value

    bounds = [(0, 1) for _ in range(num_servers)]  # Bounds for each server

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        schedule = result.x
    else:
        schedule = None
    
    return schedule, result.success

# Step 6: Create a Detailed Report
def create_pdf_report(summary_stats, model_mse, schedule, pdf_filename='Predictive Analysis for server Downtime and maintenance_report.pdf'):
    """ Generate a PDF report that includes all analysis steps, visualizations, and recommendations """
    document = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    style_heading = styles['Heading2']
    style_normal = styles['Normal']

    content = []

    # Title
    content.append(Paragraph('Maintenance Data Analysis Report', style_heading))

    # Introduction
    content.append(Paragraph('This report provides a comprehensive analysis of server maintenance data, including predictive modeling, scheduling optimization, and integration of Power BI visuals.', style_normal))
    content.append(Paragraph('The following steps are covered:', style_normal))

    # Data Extraction
    content.append(Paragraph('1. Data Extraction:', style_heading))
    content.append(Paragraph('Data was extracted from an SQLite database. The dataset includes various maintenance records, which were loaded into a pandas DataFrame for further analysis.', style_normal))

    # Data Analysis
    content.append(Paragraph('2. Data Analysis:', style_heading))
    content.append(Paragraph('The extracted data was analyzed to compute summary statistics and identify trends. The date columns were converted to datetime format to facilitate time-based analysis.', style_normal))
    content.append(Paragraph('Summary statistics of the dataset:', style_normal))
    data_summary = summary_stats.to_string()
    content.append(Paragraph(data_summary, style_normal))

    # Visualization
    content.append(Paragraph('3. Visualization:', style_heading))
    content.append(Paragraph('Visualizations were created to depict downtime duration over time and average monthly downtime duration.', style_normal))
    content.append(Paragraph('Downtime Duration Over Time:', style_normal))
    content.append(Image('downtime_over_time.png'))
    content.append(Paragraph('Average Monthly Downtime Duration:', style_normal))
    content.append(Image('Downtime.png'))

    # Predictive Maintenance Model
    content.append(Paragraph('4. Predictive Maintenance Model:', style_heading))
    content.append(Paragraph('A predictive model was developed to forecast downtime based on historical data. The model used features such as the number of days since the last maintenance and server type.', style_normal))
    content.append(Paragraph(f'Model Mean Squared Error: {model_mse:.2f}', style_normal))

    # Maintenance Scheduling Optimization
    content.append(Paragraph('5. Maintenance Scheduling Optimization:', style_heading))
    content.append(Paragraph('An optimization model was used to determine the best schedule for maintenance activities to minimize downtime and costs.', style_normal))
    if schedule is not None:
        content.append(Paragraph(f'Optimized Maintenance Schedule: {schedule}', style_normal))
    else:
        content.append(Paragraph('Optimization failed to find a feasible solution.', style_normal))

    # Power BI Integration
    content.append(Paragraph('6. Power BI Dashboard Visuals:', style_heading))
    content.append(Paragraph('The following visuals from the Power BI dashboard are included to provide additional insights:', style_normal))
    
    # Add Power BI exported visuals (update the paths accordingly)
    power_bi_images = ['Server_Performance.png', 'Downtime.png','Maintenance.png']  # Replace with actual filenames
    for img_path in power_bi_images:
        content.append(Image(img_path))

    # Recommendations
    content.append(Paragraph('7. Recommendations:', style_heading))
    recommendations = [
        "Consider increasing server maintenance frequency during peak downtime periods.",
        "Analyze the cause of downtime for different servers to identify recurring issues.",
        "Optimize maintenance schedules to reduce average downtime duration."
    ]
    for rec in recommendations:
        content.append(Paragraph(rec, style_normal))

    # Build the PDF
    document.build(content)

# Main function to run the complete process
def main():
    db_path = r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db'
    data = extract_data_from_sqlite(db_path)
    summary_stats, monthly_trends = analyze_data(data)
    generate_visualizations(data, monthly_trends)
    model, model_mse = build_predictive_model(data)
    schedule, success = optimize_maintenance_schedule(data)
    create_pdf_report(summary_stats, model_mse, schedule)

if __name__ == "__main__":
    main()

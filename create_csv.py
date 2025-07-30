import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_FILE = 'health_data.csv'

def create_health_data_csv():
    """Generates a synthetic health dataset and saves it to a CSV file."""
    logging.info(f"Generating synthetic data for {OUTPUT_FILE}...")
    
    num_rows = 2000
    locations = ['Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'West Bengal', 'Karnataka', 'Rajasthan', 'Gujarat']
    start_date, end_date = '2024-01-01', '2024-12-31'
    start, end = datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d')
    date_range_days = (end - start).days

    data = {
        'date': [start + timedelta(days=random.randint(0, date_range_days)) for _ in range(num_rows)],
        'location': random.choices(locations, k=num_rows),
        'age_group': random.choices(['18-40', '41-60', '61+'], weights=[0.3, 0.4, 0.3], k=num_rows),
        'gender': random.choices(['Male', 'Female'], weights=[0.45, 0.55], k=num_rows),
        'severity': np.random.randint(1, 11, size=num_rows),
        'cases': np.random.randint(20, 750, size=num_rows)
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Successfully created {OUTPUT_FILE}.")

if __name__ == '__main__':
    create_health_data_csv()

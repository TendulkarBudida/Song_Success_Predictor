import pandas as pd # for getting rating values from views count
import numpy as np

# Read the CSV file
df = pd.read_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv')

# Define conditions and corresponding values
conditions = [
    (df['views'] > 500000),
    (df['views'] > 400000),
    (df['views'] > 300000),
    (df['views'] > 200000),
    (df['views'] > 100000),
    (df['views'] > 0)
]

values = [5, 4, 3, 2, 1, 0]

# Add a new column based on conditions
df['rating'] = np.select(conditions, values, default=0)

# Save the modified DataFrame to a new CSV file
df.to_csv('C:\\Song_Success_Predictor\\pre_processing\\selected_columns.csv', index=False)
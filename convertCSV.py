import pandas as pd
#Read the CSV file
df = pd.read_csv('doc/sample2.csv')
# Merge columns 1 to 34 into a single column
merged_data = df.iloc[:, 0:35].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# Store the data from the 35th column in a separate column
rest_data = df.iloc[:, 35]

# Create a new DataFrame with the merged data and the rest data
new_df = pd.DataFrame({'questions': merged_data, 'className': rest_data})

# Save the new DataFrame to a new CSV file
new_df.to_csv('doc/merged_file.csv', index=False)
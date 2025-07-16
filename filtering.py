import os
import pandas as pd
from tqdm import tqdm

for file in tqdm(os.listdir(os.path.join('output', 'csv'))):
    data = pd.read_excel(os.path.join('output', 'csv', file))
    data.sort_values(by=['folder', 'filename'], inplace=True)
    # Apply IQR filtering to three classes ['L1_GT', 'L2_GT', 'S1_GT / S2_GT']

    for class_name in ['L1_GT', 'L2_GT', 'S1_GT', 'S2_GT']:
        class_data = data[data['folder'] == class_name]
        if not class_data.empty:
            Q1 = class_data['iri'].quantile(0.25)
            Q3 = class_data['iri'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter outliers
            filtered_data = class_data[(class_data['iri'] >= lower_bound) & (
                class_data['iri'] <= upper_bound)]

            data.loc[data['folder'] == class_name,
                     'filtered_iri'] = filtered_data['iri']
        else:
            print(
                f"No data found for {class_name} in {file}. Skipping filtering.")
        # Save the filtered data back to the same file
    data.to_excel(os.path.join(
        'output', 'csv_iqr', file), index=False)

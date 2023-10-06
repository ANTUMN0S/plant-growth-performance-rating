'''
We want to categorize the growth performance of basil plants with a simple scoring system (1-5).
My idea so far:
The scores will then be distributed by the following logic:
-1: Below 20th percentile
-2: Between 20th and 40th percentile
-3: Between 40th and 60th percentile
-4: Between 60th and 80th percentile
-5: Above 80th percentile
'''

# import dependencies
import pandas as pd
import numpy as np
import joblib

def get_percentile_range(plant_age=int, dataframe=pd.DataFrame):
    # filter dataframe for plant age
    plant_age_df = dataframe[dataframe['Week'] == plant_age]

    # extract weights and calculate percentiles
    weights = plant_age_df['Weight']
    percentiles = np.percentile(weights, [20,40,60,80])

    # define percentile ranges
    percentile_ranges = [(0, percentiles[0]), (percentiles[0], percentiles[1]), (percentiles[1], percentiles[2]), (percentiles[2], percentiles[3]), (percentiles[3], np.inf)]
    return percentile_ranges

# load the plant weight dataframe and create a dictionary holding the percentile ranges
df = pd.read_csv('/home/michael/Master/plant_weights.csv')
weeks = list(set(df['Week']))
percentile_ranges = {f'{week}':get_percentile_range(week, df) for week in weeks}
joblib.dump(percentile_ranges, 'full_percentile_ranges.joblib')

#print(percentile_ranges)



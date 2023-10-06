'''
Extract relevant information from your pictures and save them in a dataframe
'''

import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

# Define the directory where the segmentation pictures are stored
segmentation_dir = '/path/to/dataset'

# Create an empty DataFrame to store the extracted information
data = pd.DataFrame(columns=['Week', 'Plant', 'Angle', 'Pot_Diameter', 'Plant_Pixels'])

# Iterate over the segmentation pictures
for file_name in tqdm(os.listdir(segmentation_dir)):
    if file_name.endswith('.png'):
        # Extract information from the file name
        week = file_name.split('_')[1]
        plant, capture_angle = file_name.split('_')[3:5]

        # Read the segmentation image
        image_path = os.path.join(segmentation_dir, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Create a mask to identify pot pixels (value 1)
        pot_mask = (image_array == 1)

        # Find the x-coordinates of all pot pixels
        pot_x_coordinates = np.argwhere(pot_mask)[:, 1]

        if len(pot_x_coordinates) < 2:
            raise ValueError(f"{file_name}: At least two pot pixels are required to find the diameter of the pot.")

        # Find the two pot pixels with the minimum and maximum x-coordinates
        leftmost_x, rightmost_x = np.min(pot_x_coordinates), np.max(pot_x_coordinates)

        # Calculate the diameter (distance between the two pot pixels)
        pot_diameter = rightmost_x - leftmost_x + 1

        # Count the number of plant pixels
        plant_pixels = (image_array == 2).sum()
        if plant_pixels == 0:
            raise ValueError(f"{file_name}: Missing plant annotation.")

        # Add the extracted information to the DataFrame
        data_dict = {'Week': week, 'Plant': plant, 'Angle': capture_angle, 
                            'Pot_Diameter': pot_diameter, 'Plant_Pixels': plant_pixels}
        dict_df = pd.DataFrame([data_dict])
        data = pd.concat([data, dict_df], ignore_index=True)

# Save the DataFrame to a CSV file
data.to_csv('dataset.csv', index=False)
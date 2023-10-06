import os

def rename_images(root_dir):
    # Iterate over the directory structure
    for week_dir in os.listdir(root_dir):
        week_path = os.path.join(root_dir, week_dir)
        if not os.path.isdir(week_path):
            continue

        for plant_dir in os.listdir(week_path):
            plant_path = os.path.join(week_path, plant_dir)
            if not os.path.isdir(plant_path):
                continue

            for degree_dir in os.listdir(plant_path):
                degree_path = os.path.join(plant_path, degree_dir)
                if not os.path.isdir(degree_path):
                    continue

                # Split the degree directory name to extract angle and Z value
                angle, z_value = degree_dir.split('_')[0], degree_dir.split('_')[1]

                # Iterate over the image files in the degree directory
                capture_angles = [0, 60, 120, 180, 240, 300]
                for i, file_name in enumerate(os.listdir(degree_path)):
                    if file_name.endswith('.jpg') or file_name.endswith('.png'):
                        file_path = os.path.join(degree_path, file_name)

                        # Generate the new file name with week, plant, angle, and Z information
                        capture_angle = capture_angles[i]
                        new_file_name = f"{week_dir}_{plant_dir}_{angle}_{capture_angle}.jpg"

                        # Rename the image file
                        new_file_path = os.path.join(degree_path, new_file_name)
                        os.rename(file_path, new_file_path)

root_directory = './Datasets/Dataset_Temp'
rename_images(root_directory)

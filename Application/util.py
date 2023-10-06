import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import joblib
import sys
import warnings
from segment_anything import SamPredictor, sam_model_registry
from sklearn.linear_model import LinearRegression
from lang_sam import LangSAM

warnings.filterwarnings("ignore") 

# set SAM variables
sys.path.append("..")
SAM_MODELS = {
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"}
sam_type = "vit_l"
checkpoint_url = SAM_MODELS[sam_type]
try:
    sam = sam_model_registry[sam_type]()
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    sam.load_state_dict(state_dict, strict=True)
except:
    raise ValueError(f"Problem loading SAM please make sure you have the right model type: {sam_type} \
        and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
        re-downloading it.")
sam.to(device='cpu')
predictor = SamPredictor(sam)

def segmentation(image):
    '''
    This function is intended to perform image-segmentation with a custom trained DeepLabV3 model, but as our model does not perform well enough,
    it is currently unused
    '''
    # setup model
    model = torchvision.models.segmentation.DeepLabV3()
    model.load_state_dict(torch.load('path/to/training_output/finetuned_weights.pth'))
    model.eval()

    # preprocess picture
    input_image = image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # pass the resulting mask as an image
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    result = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    

    return result

def segmentation_lang(image):
    # clear cache and load model
    torch.cuda.empty_cache()
    model = LangSAM()

    # first segment the pot
    text_prompt = "gray pot"
    #text_prompt = "center gray pot"
    masks, boxes, phrases, logits = model.predict(image, text_prompt)
    image_1 = masks.numpy().transpose(1,2,0)
    image_1 = image_1[:,:,0]
    # convert from bool to 0 and 1
    number_1 = np.array(image_1, dtype = int)

    # now segment the plant
    torch.cuda.empty_cache()
    text_prompt = "leafs"
    #text_prompt = "leafs of plant in the center"
    masks, boxes, phrases, logits = model.predict(image, text_prompt)
    image_2 = masks.numpy().transpose(1,2,0)
    image_2 = image_2[:,:,0]
    # convert from bool to 0 and 2
    number_2 = np.where(image_2, 2, 0)

    #merge the two arrays
    merged_array = np.maximum(number_1, number_2)
    
    # flatten it to 2D
    squeezed_array = np.squeeze(merged_array)

    return squeezed_array

def segmentation_by_coordinates(coords, image):
    image = np.asarray(image)
    labels = np.array([1])
    coords_1 = np.array([coords[0]])
    coords_2 = np.array([coords[1]])
    predictor.set_image(image)
    pot_mask, pot_scores, pot_logits = predictor.predict(
        point_coords=coords_1,
        point_labels = labels,
        multimask_output= False
        )
    plant_mask, plant_scores, plant_logits = predictor.predict(
        point_coords = coords_2,
        point_labels = labels,
        multimask_output= False
        )
    # convert from bool to numbers and merge
    pot_mask = np.where(pot_mask, 1, 0)
    plant_mask = np.where(plant_mask, 2, 0)
    merged_mask = np.maximum(pot_mask, plant_mask).squeeze()
    return merged_mask

def make_dataframe(segmentation_mask, file_path):
    # if it's a picture from the dataset, we can get the angle from there
    file_name = os.path.basename(file_path)
    # Extract information from the file name
    week = file_name.split('_')[1]
    capture_angle = file_name.split('_')[4]
    if ".jpg" in capture_angle:
        capture_angle = capture_angle.replace(".jpg", "")
    elif ".png" in capture_angle:
        capture_angle = capture_angle.replace(".png", "")  

    #convert the image to a NumPy array
    image_array = np.array(segmentation_mask)

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
        
    # Create an empty DataFrame to store the extracted information
    data_dict = {'Week': week, 'Angle': capture_angle, 'Pot_Diameter': pot_diameter, 'Plant_Pixels': plant_pixels}
    df = pd.DataFrame([data_dict])
    df['pixel_ratio'] = [32/i for i in df['Pot_Diameter']]
    df['plant_area'] = df['Plant_Pixels'] * df['pixel_ratio'] * df['pixel_ratio']
        
    return df

def make_dataframe_from_array(image_array, file_path):
    # if it's a picture from the dataset, we can get the angle from there
    file_name = os.path.basename(file_path)
    # Extract information from the file name
    week = file_name.split('_')[1]
    capture_angle = file_name.split('_')[4]
    if ".jpg" in capture_angle:
        capture_angle = capture_angle.replace(".jpg", "")
    elif ".png" in capture_angle:
        capture_angle = capture_angle.replace(".png", "")

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
        
    # Create an empty DataFrame to store the extracted information
    data_dict = {'Week': week, 'Angle': capture_angle, 'Pot_Diameter': pot_diameter, 'Plant_Pixels': plant_pixels}
    df = pd.DataFrame([data_dict])
    df['pixel_ratio'] = [32/i for i in df['Pot_Diameter']]
    df['plant_area'] = df['Plant_Pixels'] * df['pixel_ratio'] * df['pixel_ratio']
        
    return df


def weight_estimation(df):
    # import the linear regression model and percentile ranges
    regression_model = joblib.load('./sam_lr_model.joblib')
    percentile_ranges = joblib.load('./full_percentile_ranges.joblib')

    # estimate the weight
    weight = round(regression_model.predict(df[['plant_area', 'Week', 'Angle']])[0], 1)

    # get percentiles of the right week and find score
    week = str(df["Week"].iloc[0])
    percentiles = percentile_ranges[week]
    rating = [1, 2, 3, 4, 5]
    
    # assign score to the plant weight
    for i, (lower, upper) in enumerate(percentiles):
        if lower <= weight < upper:
            score = rating[i]
    
    return weight, score
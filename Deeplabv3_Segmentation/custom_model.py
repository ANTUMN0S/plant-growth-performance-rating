import torchvision
from torchvision import models
import torch
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from pick import pick
import os

class DeepLabV3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)['out']
        return output

def initialize_model(num_classes, backbone, keep_feature_extract=False, use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    weights_folder = '/path/to/Deeplabv3_Segmentation/training_output'  # Specify the path to your weights folder
    weights_paths = []
    weights_filenames = ['Use pretrained backbone']

    if os.path.exists(weights_folder) and os.path.isdir(weights_folder):
        # Iterate over files in the weights folder
        for file_name in os.listdir(weights_folder):
            file_path = os.path.join(weights_folder, file_name)
            if os.path.isfile(file_path) and backbone in file_name:
                weights_paths.append(file_path)
                weights_filenames.append(file_name)

    # Let the user select the weights the want to train with
    question = 'Please choose the weights you want your backbone to have below:'
    selected_filename, index = pick(weights_filenames, question)


    if backbone == 'resnet101':
        if selected_filename == 'Use pretrained backbone':
            backbone_weights = DeepLabV3_ResNet101_Weights
        else:
            # Find the index of the selected filename in the weights_filenames list and retrive corresponding weights
            index = weights_filenames.index(selected_filename)
            selected_path = weights_paths[index]
            backbone_weights = torch.load(selected_path)

        num_channels = 2048

    if backbone == 'mobilenet_v3_large':
        if selected_filename == 'Use pretrained backbone':
            backbone_weights = DeepLabV3_MobileNet_V3_Large_Weights
        else:
            # Find the index of the selected filename in the weights_filenames list and retrive corresponding weights
            index = weights_filenames.index(selected_filename)
            selected_path = weights_paths[index]
            backbone_weights = torch.load(selected_path)
        num_channels = 960

    model_backbone = getattr(torchvision.models.segmentation, f'deeplabv3_{backbone}')
    model_deeplabv3 = model_backbone(weights = backbone_weights)
    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    input_size = 224

    
    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(num_channels, num_classes)

    return model_deeplabv3, input_size


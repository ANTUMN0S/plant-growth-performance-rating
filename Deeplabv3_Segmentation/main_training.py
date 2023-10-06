import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
import pathlib
from pick import pick
from datetime import date

# Local import
from dataloader import DataLoaderSegmentation
from custom_model import initialize_model
from train import train_model
import matplotlib.pyplot as plt

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

"""
    Version requirements:
        PyTorch Version:  1.4.0
        Torchvision Version:  0.5.0
"""


def main(data_dir, dest_dir, num_classes, batch_size, num_epochs, keep_feature_extract, weight):
# def main():

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: DataLoaderSegmentation(os.path.join(data_dir, x), x) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Let user pick the backbone
    question = 'Please choose the backbone you want to train your CNN with below:'
    possibilities = ['resnet101', 'mobilenet_v3_large']
    backbone, index = pick(possibilities, question)

    print("Initializing Model...")

    # Initialize model
    model_deeplabv3, input_size = initialize_model(num_classes, backbone, keep_feature_extract, use_pretrained=True)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_deeplabv3 = model_deeplabv3.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_deeplabv3.parameters()
    print("Params to learn:")
    if keep_feature_extract:
        params_to_update = []
        for name, param in model_deeplabv3.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_deeplabv3.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Observe that all parameters are being optimized
    learning_rate = 0.001
    momentum = 0.9
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(device) if weight else None))

    # Prepare output directory
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Train...")

    # Train and evaluate
    model_deeplabv3_state_dict, hist, train_acc, train_loss, val_acc, val_loss = train_model(model_deeplabv3, num_classes, dataloaders_dict, criterion, optimizer_ft, device, dest_dir, num_epochs=num_epochs)

    # get the current date
    today = date.today().strftime("%Y_%m_%d")
    print("Save ...")
    torch.save(model_deeplabv3_state_dict, os.path.join(dest_dir, f"best_DeepLabV3_{backbone}_{today}.pth"))

    # plot the values for accuracy and loss for both train and val
    # Create figure and axis objects
    fig, (train_ax, val_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Train Data
    train_epochs = list(range(1, len(train_loss) + 1))
    train_ax.plot(train_epochs, train_loss, label='Train Loss', color='blue')
    train_ax.plot(train_epochs, train_acc, label='Train Accuracy', color='orange')

    # Plot Validation Data
    val_epochs = list(range(1, len(val_loss) + 1))
    val_ax.plot(val_epochs, val_loss, label='Validation Loss', color='green')
    val_ax.plot(val_epochs, val_acc, label='Validation Accuracy', color='red')

    # Set Titles and Labels
    train_ax.set_title('Train Loss and Accuracy')
    val_ax.set_title('Validation Loss and Accuracy')
    train_ax.set_ylabel('Loss/Accuracy')
    val_ax.set_ylabel('Loss/Accuracy')
    plt.xlabel('Epochs')

    # Add Legends
    train_ax.legend(loc='upper right')
    val_ax.legend(loc='upper right')
    fig.suptitle(f'Epochs:{num_epochs}, Batchsize: {batch_size}, Learning Rate: {learning_rate}, Momentum: {momentum}')

    # Show the plot
    plt.tight_layout()
    plt.show()


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help='Specify the dataset directory path, should contain train/Images, train/Labels, val/Images and val/Labels')
    parser.add_argument(
        "dest_dir", help='Specify the  directory where model weights shall be stored.')
    parser.add_argument("--num_classes", default=3, type=int, help="Number of classes in the dataset, index 0 for no-label should be included in the count")
    parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training (change depending on how much memory you have)")
    parser.add_argument("--keep_feature_extract", action="store_true", help="Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params")
    parser.add_argument('-w', action='append', type=float, help="Add more weight to some classes. If this argument is used, then it should be called as many times as there are classes (see --num_classes)")

    args = parser.parse_args()

    # Build weight list
    weight = []
    if args.w:
        for w in args.w:
            weight.append(w)

    main(args.data_dir, args.dest_dir, args.num_classes, args.batch_size, args.epochs, args.keep_feature_extract, weight)


if __name__ == '__main__':
    args_preprocess()
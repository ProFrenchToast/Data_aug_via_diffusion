import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import argparse


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = ndarr.astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size,args.image_size)),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(output_dir, run_name):
    model_dir = os.path.join(output_dir,"models")
    results_dir = os.path.join(output_dir,"results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, run_name), exist_ok=True)
    os.makedirs(os.path.join(results_dir, run_name), exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser()
    # Add the needed Arguments, run name and dataset path are required
    parser.add_argument("--run_name", help="Provide the name of the run.", default="No_Name", required=True)
    parser.add_argument("--epochs", help="Specifies the number of epochs to run.", default=100)
    parser.add_argument("--batch_size", help="Specifies number of samples in each batch during training.", default=2)
    parser.add_argument("--img_size", help="Specifiies the size the images are transformed to", default=128)
    parser.add_argument("--dataset_path", help="Specifiies path to the dataset to train on", required=True)
    parser.add_argument("--lr", help="Specifiies learning rate for training", default=1e-3)
    parser.add_argument("--model", help="Specifies what the architecture of the model will be", default="ComplexUnet")
    parser.add_argument("--output_dir", help="the directory in which to save the run data to", default=os.curdir)
    parser.add_argument("--resume_training", help="If set then continue training from the given epoch", default=None)
    parser.add_argument("--ema", help="The discounting rate of the exponential moving average", default=0.995)
    return parser

def get_generator_args_parser():
    parser = argparse.ArgumentParser()
    # add the argument for the generator 
    parser.add_argument("--model_path", help="Path to the model used to generate new samples", required=True)
    parser.add_argument("--output_path", help="The path to save the generated samples to", required=True)
    parser.add_argument("--num_demos", help="The number of demos to generate", default=1000)
    parser.add_argument("--samples_per_demo", help="The number of samples in each demo, ie the length", default=1)
    parser.add_argument("--num_steps", help="The number of diffusion steps taken when generating samples", default=1000)
    parser.add_argument("--model", help="Specifies what the architecture of the model will be", default="ComplexUnet")
    return parser

def get_rollout_args_parser():
    parser = argparse.ArgumentParser()
    # add the argument for the rollout code
    parser.add_argument("--model_weights", help="The path to the weights used to generate the rollouts", required=True)
    parser.add_argument("--env", help="The env in which to rollout the model", required=True)
    parser.add_argument("--output_dir", help="the directory to save the video outputs to", required=True)
    parser.add_argument("--seeds", help="The list of seeds to initalise the env with", nargs='+', required=True)
    parser.add_argument("--max_steps", help="maximum number fo steps in the rollout", default=5000)
    parser.add_argument("--device", help="te device on whice to run the policy", default="cpu")
    parser.add_argument("--model_file", help="The path to the model definition used to generate the rollouts", default=None)
    return parser

def get_prob_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", help="the path to the model used to estimate the probabilities", required=True)
    parser.add_argument("--data_dir", help="The directory containing the demonstrations to estimate", required=True)
    return parser

def action_Dict2Vec(action):
    # takes an action dictionary and converts it to a vector for model consumption
    vec = np.zeros(25, dtype=np.float32)
    # this isnt the most sustainable but it avaoids any weirdness with the ordering
    vec[0] = action["ESC"]
    vec[1] = action["back"]
    vec[2] = action["drop"]
    vec[3] = action["forward"]
    vec[4] = action["hotbar.1"]
    vec[5] = action["hotbar.2"]
    vec[6] = action["hotbar.3"]
    vec[7] = action["hotbar.4"]
    vec[8] = action["hotbar.5"]
    vec[9] = action["hotbar.6"]
    vec[10] = action["hotbar.7"]
    vec[11] = action["hotbar.8"]
    vec[12] = action["hotbar.9"]
    vec[13] = action["inventory"]
    vec[14] = action["jump"]
    vec[15] = action["left"]
    vec[16] = action["right"]
    vec[17] = action["sneak"]
    vec[18] = action["sprint"]
    vec[19] = action["swapHands"]

    vec[20] = action["camera"][0]
    vec[21] = action["camera"][1]

    vec[22] = action["attack"]
    vec[23] = action["use"]
    vec[24] = action["pickItem"]
    return vec



def action_Vec2Dict(action):
    # takes an action vector and converts it to a dictionary for env consumption
    Dict = {
        "ESC": action[0],
        "back": action[1],
        "drop": action[2],
        "forward": action[3],
        "hotbar.1": action[4],
        "hotbar.2": action[5],
        "hotbar.3": action[6],
        "hotbar.4": action[7],
        "hotbar.5": action[8],
        "hotbar.6": action[9],
        "hotbar.7": action[10],
        "hotbar.8": action[11],
        "hotbar.9": action[12],
        "inventory": action[13],
        "jump": action[14],
        "left": action[15],
        "right": action[16],
        "sneak": action[17],
        "sprint": action[18],
        "swapHands": action[19],
        "camera": np.array([action[20], action[21]]),
        "attack": action[22],
        "use": action[23],
        "pickItem": action[24],
    }
    return Dict

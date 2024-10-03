#given a diffusion model and the arguments make a new dataset of 

import json
import os
import cv2
import numpy as np
import torch
from data_loader import env_action_to_json_action
from forward_process import Sde
from modules import simpleUnetAction, stateActionUnet
from utils import action_Vec2Dict, get_generator_args_parser

def save_video(obs, output_file):
    channels, height, width = obs[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    fps = 20  # Specify the frames per second
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each tensor, convert it to an image, and write it to the video file
    for i, ob in enumerate(obs):
        image = ob.cpu()[0]
        image = image.permute(1, 2, 0)
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # needed for the colours to match
        video_writer.write(image)
    video_writer.release()

def save_actions(actions, outputfile):
    #for each action convert it to json version then to a string 
    output_string = ""
    for action in actions:
        dict_action = action_Vec2Dict(action[0].cpu())
        json_action = env_action_to_json_action(dict_action)
        action_string = json.dumps(json_action)
        output_string += action_string + "\n"

    json_file = open(outputfile, "w")
    json_file.write(output_string)
    json_file.close()

    


def generate_dataset(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_size = 25
    if args.model == "SimpleUnet":
        model = simpleUnetAction(action_size).to(device)
    elif args.model == "ComplexUnet":
        model = stateActionUnet(action_size, device=device).to(device)
    model.load_state_dict(torch.load(args.model_path))
    ForwardProcess = Sde(img_size=128, action_size=model.action_size, device=device)

    
    os.makedirs(args.output_path, exist_ok=True)


    for demo in range(args.num_demos):
        # make the video and action array
        obs = []
        actions = []

        for sample in range(args.samples_per_demo):
            # generate the sample for the demo using the model
            new_obs, new_action = ForwardProcess.sample(model, num_steps=args.num_steps)
            obs.append(new_obs)
            actions.append(new_action)
            # reset the forward process to sample from another point in the dist
            ForwardProcess.reset() 

        # now save the states as a video and the actions as a json file
        video_path = os.path.join(args.output_path, f"{demo}.mp4")
        action_path = os.path.join(args.output_path, f"{demo}.jsonl")
        save_video(obs, video_path)
        save_actions(actions, action_path)



def main():
    parser = get_generator_args_parser()
    args = parser.parse_args()
    args.num_demos = int(args.num_demos)
    args.samples_per_demo = int(args.samples_per_demo)
    args.num_steps = int(args.num_steps)
    generate_dataset(args)

if __name__ == '__main__':
    main()
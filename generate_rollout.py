import json
import os
import pickle
import gym
import cv2
import numpy as np
import torch
from data_loader import env_action_to_json_action
from utils import get_rollout_args_parser
from vpt_lib.agent import MineRLAgent

def save_video(obs, output_file):
    height, width, channels = obs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    fps = 20  # Specify the frames per second
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each tensor, convert it to an image, and write it to the video file
    for i, ob in enumerate(obs):
        image = ob
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # needed for the colours 
        video_writer.write(image)
    video_writer.release()

def save_actions(actions, outputfile):
    #for each action convert it to json version then to a string 
    output_string = ""
    for action in actions:
        json_action = env_action_to_json_action(action)
        action_string = json.dumps(json_action)
        output_string += action_string + "\n"

    json_file = open(outputfile, "w")
    json_file.write(output_string)
    json_file.close()

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def generate_rollout(player, env, seed, max_steps):
    all_obs = []
    all_actions = []

    env.seed(seed)
    obs = env.reset()
    player.reset()
    all_obs.append(torch.from_numpy(obs["pov"].copy()))
    done = False
    steps = 0

    while not done:
        # might need to tweek this based on how the player is implemented
        action = player.get_action(obs)
        obs, _, done, _ = env.step(action)
        all_obs.append(torch.from_numpy(obs["pov"].copy()))
        all_actions.append(action)
        steps +=1

        if steps >= max_steps:
            print(f"breaking loop {steps} {max_steps}")
            break

    #need to pop the last observation so there are the same number of actions and obs
    final_ob = all_obs.pop()
    return all_obs, all_actions

def generate_videos(args):
    if args.model_file is not None:
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(args.model_file)
    else:
        agent_pi_head_kwargs = None
        agent_policy_kwargs = None
    agent = MineRLAgent(device=args.device,
                        policy_kwargs=agent_policy_kwargs,
                        pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(args.model_weights)
    env = gym.make(args.env)
    env.reset()

    output_folder = os.path.join(args.output_dir, args.env)
    os.makedirs(output_folder, exist_ok=True)

    for seed in args.seeds:
        seed = int(seed)
        obs, actions = generate_rollout(agent, env, seed, args.max_steps)
        video_path = os.path.join(output_folder, f"{seed}.mp4")
        action_path = os.path.join(output_folder, f"{seed}.jsonl")
        save_video(obs, video_path)
        save_actions(actions, action_path)







def main():
    #get and setup the args here
    parser = get_rollout_args_parser()
    args = parser.parse_args()
    generate_videos(args)


if __name__ == '__main__':
    main()
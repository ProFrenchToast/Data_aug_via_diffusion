



# this code estimates the log probability of generating a given 
# state action pair using a trained diffusion model
# it outputs a json file with the estimate for each frame
import json
import os

import numpy as np
from scipy import integrate
import torch
from data_loader import Data_Loader
from forward_process import Sde, combine_state_action, ode_sampler
from modules import stateActionUnet
from utils import action_Dict2Vec, get_prob_args_parser





def calculate_log_probability(model, sampler, images, actions):
    action_vec = torch.from_numpy(action_Dict2Vec(actions[0]))
    image_vec = torch.from_numpy(np.stack(images, axis=0))
    combined = combine_state_action(image_vec, action_vec)
    z, prob = sampler.ode_likelihood(combined, model, device="cuda")
    return prob


def main():
    parser = get_prob_args_parser()
    args = parser.parse_args()

    action_size = 25
    device = "cuda"
    model = stateActionUnet(action_size, device=device).to(device)
    model.load_state_dict(torch.load(args.model_path))
    forward_process = Sde(device=device)
    sampler = ode_sampler(forward_process.marginal_prob_std, forward_process.diffusion_coeff, device=device)
    dataloader = Data_Loader(args.data_dir, n_workers=1, batch_size=1)
    trajectory_probs = {}
    current_traj = None

    for i, (images, actions, batch_episode_id) in enumerate(dataloader):
        # first check to see if a new trajectory
        if current_traj != batch_episode_id[0]:
            #if so then add a new entry in the dict
            current_traj = batch_episode_id[0]
            trajectory_probs[current_traj] = []

        # now calculate the probability of this sample
        probs = calculate_log_probability(model, sampler, images, actions)
        print(probs)
        trajectory_probs[current_traj].append(probs)

    # finally save the probs to a json file
    output_string = json.dumps(trajectory_probs)
    output_file  = os.path.join(args.data_dir, "probabilities.json")
    json_file = open(output_file, "w")
    json_file.write(output_string)
    json_file.close()





if __name__ == '__main__':
    main()



import copy
import os
from data_loader import Data_Loader # needs to be imported before the tensorboard for some godforsaken reason
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from forward_process import Sde
from utils import *
from modules import EMA, simpleUnetAction, stateActionUnet
import logging

from torch.utils.tensorboard import SummaryWriter
import numpy as np


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")





def train(args):
    setup_logging(args.output_dir, args.run_name)
    device = args.device
    action_size = 25
    if args.model == "SimpleUnet":
        model = simpleUnetAction(action_size).to(device)
    elif args.model == "ComplexUnet":
        model = stateActionUnet(action_size, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    ForwardProcess = Sde(img_size=args.image_size, action_size=model.action_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    num_samples = None

    if args.resume_training is not None:
        split_list = args.resume_training.split("-")
        start_epoch = int(split_list[0])
        model_path = os.path.join(os.path.join("models", args.run_name), f"ckpt-ema-{args.resume_training}.pt")
        model.load_state_dict(torch.load(model_path))
        start_epoch += 1
        print(f"loaded model from {model_path}")
    else:
        start_epoch = 0

    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    print("starting training")
    for epoch in range(start_epoch, args.epochs):
        dataloader = Data_Loader(args.dataset_path, n_workers=args.batch_size, batch_size=args.batch_size)
        l = len(dataloader.unique_ids)
        mid_epoch_sample = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader, total=num_samples)
        for i, (images, actions, batch_episode_id) in enumerate(pbar):

            if i % 40000 == 0:
                sampled_images, sampled_actions = ForwardProcess.sample(model)
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}-{mid_epoch_sample}.jpg"))
                ema_images, ema_actions = ForwardProcess.sample(ema_model)
                save_images(ema_images, os.path.join("results", args.run_name, f"ema-{epoch}-{mid_epoch_sample}.jpg"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ckpt-ema-{epoch}-{mid_epoch_sample}.pt"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt-{epoch}-{mid_epoch_sample}.pt"))
                mid_epoch_sample +=1

            images = np.stack(images, axis=0)
            images = torch.from_numpy(images)
            images = images.to(device)
            
            # convert the actions to vectors 
            action_vecs = []
            for action in actions:
                action_vecs.append(action_Dict2Vec(action))
            action_vecs = np.array(action_vecs)
            action_vecs = torch.from_numpy(action_vecs)
            action_vecs = action_vecs.to(device)

            loss = ForwardProcess.loss_fn(model, images, action_vecs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        num_samples = i+1
        dataloader.__del__()
        sampled_images, sampled_actions = ForwardProcess.sample(model)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt-{epoch}.pt"))


def launch():
    parser = get_args_parser()

    args = parser.parse_args()
    #args.run_name = "DDPM_128-lr1e-3"
    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.image_size = args.img_size
    #args.dataset_path = r"./Datasets/Landscapes"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = np.float32(args.lr)
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()

import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.latent import UncondLatentDataset
from models.split_vae import VAE
from util import configure_device, get_dataset, save_as_images


@click.group()
def cli():
    pass


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


# TODO: Upgrade the commands in this script to use hydra config
# and support Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--dataset", default="celebamaskhq")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def reconstruct(
    chkpt_path,
    root,
    device="gpu:0",
    dataset="celebamaskhq",
    image_size=128,
    num_samples=-1,
    save_path=os.getcwd(),
    write_mode="image",
):
    # dev, _ = configure_device(device) # GPUs
    dev = configure_device(device)
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")

    # Dataset
    dataset = get_dataset(dataset, root, image_size, norm=False, flip=False)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward_recons(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :].cpu())
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch.cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0)
    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
        save_as_images(
            cat_img,
            file_name=os.path.join(save_path, "orig"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.argument("img-root")
@click.option("--seed", default=3214, type=int)
@click.option("--device", default="gpu:1")
@click.option("--dataset", default="ffhq")
@click.option("--vary", default="z_local")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample_vary_z(
    z_dim,
    chkpt_path,
    img_root,
    seed=20,
    dataset="ffhq",
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
    vary="z_local"
):
    seed_everything(seed)
    # dev, _ = configure_device(device) # GPUs sampling
    dev= configure_device(device) # CPU sampling
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")

    # Image dataset
    img_d_type = dataset
    img_dataset = get_dataset(img_d_type, img_root, image_size, norm=False, flip=False)
    # Image Loader
    img_loader = DataLoader(
        img_dataset,
        1,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )


    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()
    for img_idx, img_batch in tqdm(enumerate(img_loader)):
        seed_everything(img_idx)
        if img_idx == 30:
            break
        sample_list = []
        orig_list = []
        count = 0
        if vary == "z_local":
            mu_fixed, logvar_fixed = vae.encode(img_batch)
        elif vary == "z_global":
            img_batch_aux, _ = vae.scramble(img_batch) 
            mu_fixed, logvar_fixed = vae.encode_aux(img_batch_aux)

        z_fixed = vae.reparameterize(mu_fixed, logvar_fixed)
        ori_recons = vae.forward_recons(img_batch)
        sample_list.append(ori_recons.cpu())
        orig_list.append(img_batch.cpu())
        # Latent dataset
        dataset = UncondLatentDataset((5000, z_dim, 1, 1))
        # Latent Loader
        loader = DataLoader(
            dataset,
            1,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        for _, batch_temp in tqdm(enumerate(loader)):
            batch_temp = batch_temp.to(dev)

            if vary == "z_local":
                batch = torch.cat((z_fixed, batch_temp), dim=1)
            elif vary == "z_global":
                batch = torch.cat((batch_temp, z_fixed), dim=1)

            with torch.no_grad():
                recons = vae.forward(batch)

            if count + recons.size(0) >= num_samples and num_samples != -1:
                sample_list.append(recons[:num_samples, :, :, :].cpu())
                break

            # Not transferring to CPU leads to memory overflow in GPU!
            sample_list.append(recons.cpu())
            count += recons.size(0)

        cat_sample = torch.cat(sample_list, dim=0)
        cat_orig = torch.cat(orig_list, dim=0)
        # Save the image and reconstructions as numpy arrays
        subfolder_path = os.path.join(save_path, str(img_idx))
        os.makedirs(subfolder_path, exist_ok=True)

        if write_mode == "image":
            save_as_images(
                cat_sample,
                file_name=os.path.join(subfolder_path, "vae_img_{}".format(img_idx)),
                denorm=False,
            )
            save_as_images(
                cat_orig,
                file_name=os.path.join(subfolder_path, "orig_img_{}".format(img_idx)),
                denorm=False,
            )            
        else:
            np.save(os.path.join(subfolder_path, "recons.npy"), cat_sample.numpy())


@cli.command()
@click.argument("chkpt-path")
@click.argument("img-root")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
def style_transfer(
    chkpt_path,
    img_root,
    seed=0,
    device="gpu:0",
    image_size=128,
    save_path=os.getcwd(),
):
    seed_everything(seed)
    # dev, _ = configure_device(device) # GPUs sampling
    dev= configure_device(device) # CPU sampling


    img_root_global = os.path.join(img_root, "global")
    img_root_local = os.path.join(img_root, "local")


    # Global Image Dataset
    img_d_type = "ffhq"
    img_dataset1 = get_dataset(img_d_type, img_root_global, image_size, norm=False, flip=False)
    # Image Loader
    img_loader1 = DataLoader(
        img_dataset1,
        1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Local Image Dataset
    img_dataset2 = get_dataset(img_d_type, img_root_local, image_size, norm=False, flip=False)
    # Image Loader
    img_loader2 = DataLoader(
        img_dataset2,
        1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()
    os.mkdir(save_path)
    save_path_local =  os.path.join(save_path, "local")
    save_path_global =  os.path.join(save_path, "global")
    os.mkdir(save_path_local)
    os.mkdir(save_path_global)
    for img_idx1, img_batch1 in tqdm(enumerate(img_loader1)):
        # if img_idx == 20:
        #     break
        
        count = 0

        # Extract z_global
        mu_1, logvar_1 = vae.encode(img_batch1)
        z_global = vae.reparameterize(mu_1, logvar_1)

        ori_recons_global = vae.forward_recons(img_batch1)
        
        
        for img_idx2, img_batch2 in tqdm(enumerate(img_loader2)):
            img_batch2 = img_batch2.to(dev)

            # Extract z_local
            img_batch2_aux, _ = vae.scramble(img_batch2) 
            mu_2, logvar_2 = vae.encode_aux(img_batch2_aux)
            z_local = vae.reparameterize(mu_2, logvar_2)

            
            ori_recons_local = vae.forward_recons(img_batch2)

            batch = torch.cat((z_global, z_local), dim=1)

            with torch.no_grad():
                recons = vae.forward(batch)

            # Not transferring to CPU leads to memory overflow in GPU!
            count += recons.size(0)



            # Save the image and reconstructions as numpy arrays
            
            sub_path = os.path.join(save_path, "g{}l{}".format(img_idx1, img_idx2))

            os.makedirs(sub_path, exist_ok=True)
            save_as_images(
                ori_recons_local.cpu(),
                file_name=os.path.join(save_path_local, "ori_img_local_{}".format(img_idx2)),
                denorm=False,
            )
            save_as_images(
                recons.cpu(),
                file_name=os.path.join(sub_path, "vae_img_{}_{}".format(img_idx1, img_idx2)),
                denorm=False,
            )

        save_as_images(
        ori_recons_global.cpu(),
        file_name=os.path.join(save_path_global, "ori_img_global_{}".format(img_idx1)),
        denorm=False,
    )        




@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
):
    seed_everything(seed)
    # dev, _ = configure_device(device) # GPUs sampling
    dev= configure_device(device) # CPU sampling
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")

    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1))

    # Loader
    loader = DataLoader(
        dataset,
        2,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())




if __name__ == "__main__":
    cli()

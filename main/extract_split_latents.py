import click
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from util import configure_device, get_dataset
from models.split_vae import VAE


@click.group()
def cli():
    pass


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:0")
@click.option("--dataset-name", default="cifar10")
@click.option("--image-size", default=32)
@click.option("--save-path", default=os.getcwd())
def extract(
    vae_chkpt_path,
    root,
    device="gpu:0",
    dataset_name="cifar10",
    image_size=32,
    save_path=os.getcwd(),
):
    # dev, _ = configure_device(device)
    dev = 'cuda'

    # Dataset
    dataset = get_dataset(dataset_name, root, image_size, norm=False, flip=False)

    # Loader
    loader = DataLoader(
        dataset,
        64,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Load VAE
    vae = VAE.load_from_checkpoint(vae_chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    z_list = []
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            batch_aux, _ = vae.scramble(batch)
            mu, logvar = vae.encode(batch)
            mu_aux, logvar_aux = vae.encode_aux(batch_aux)

            z_main = vae.reparameterize(mu, logvar)
            z_aux = vae.reparameterize(mu_aux, logvar_aux)

            z = torch.cat((z_main, z_aux), dim=1)

        # Not transferring to CPU leads to memory overflow in GPU!
        z_list.append(z.cpu())

    cat_z = torch.cat(z_list, dim=0)

    # Save the latents as numpy array
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f"latents_{dataset_name}.npy"), cat_z.numpy())


if __name__ == "__main__":
    cli()

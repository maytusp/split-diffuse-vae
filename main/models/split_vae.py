import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class Encoder(nn.Module):
    def __init__(self, block_config_str, channel_config_str):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, down_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue
            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)

        # Latents
        self.mu = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)
        self.logvar = nn.Conv2d(channel_config[1], channel_config[1], 1, bias=False)

    def forward(self, input):
        x = self.in_conv(input)
        x = self.block_mod(x)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self, input_res, block_config_str, channel_config_str):
        super().__init__()
        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)
        self.last_conv = nn.Conv2d(channel_config[input_res], 3, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.block_mod(input)
        x = self.last_conv(x)
        return torch.sigmoid(x)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):
    def __init__(
        self,
        input_res,
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        aux_enc_block_str,
        aux_dec_block_str,
        aux_enc_channel_str,
        aux_dec_channel_str,        
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str

        self.aux_enc_block_str = aux_enc_block_str
        self.aux_dec_block_str = aux_dec_block_str
        self.aux_enc_channel_str = aux_enc_channel_str
        self.aux_dec_channel_str = aux_dec_channel_str

        self.alpha = alpha
        self.lr = lr

        # Encoder architecture
        self.enc = Encoder(self.enc_block_str, self.enc_channel_str)

        # Auxiliary Encoder architecture
        self.enc_aux = Encoder(self.aux_enc_block_str, self.aux_enc_channel_str)

        # Decoder Architecture
        self.dec = Decoder(self.input_res, self.dec_block_str, self.dec_channel_str)

        # Auxiliary Decoder architecture
        self.dec_aux = Decoder(self.input_res, self.aux_dec_block_str, self.aux_dec_channel_str)

    def encode(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def encode_aux(self, x_aux):
        mu_aux, logvar_aux = self.enc_aux(x_aux)
        return mu_aux, logvar_aux

    def decode(self, z):
        return self.dec(z)

    def decode_aux(self, z_aux):
        return self.dec_aux(z_aux)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):
        # For generating reconstructions during inference
        x_aux, _ = self.scramble(x)

        mu, logvar = self.encode(x)
        mu_aux, logvar_aux = self.encode_aux(x_aux)

        z = self.reparameterize(mu, logvar)
        z_aux = self.reparameterize(mu_aux, logvar_aux)
        z_total = torch.cat((z,z_aux), dim=1)

        decoder_out = self.decode(z_total)
        return decoder_out

    def scramble(self, img_batch, patch_size=16):
        bs, ch, w, h = img_batch.shape
        patches = img_batch.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patch_dim_0 = patches.shape[2]
        patch_dim_1 = patches.shape[3]
        r = torch.randperm(patch_dim_0)
        c = torch.randperm(patch_dim_1)
        scramble_patches = patches[:, :,r[:, None], c, :, :, :]
        scramble_img_batch = torch.reshape(torch.swapaxes(torch.squeeze(torch.swapaxes(scramble_patches, 1, 4), dim=4), 3,4)
                                            , shape=(bs, ch, w, h))
        return scramble_img_batch, scramble_patches

    def training_step(self, batch, batch_idx):
        x = batch
        x_aux, _ = self.scramble(batch)

        # Encoder
        mu, logvar = self.encode(x)
        mu_aux, logvar_aux = self.encode_aux(x_aux)


        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)
        z_aux = self.reparameterize(mu_aux, logvar_aux)

        z_total = torch.cat((z,z_aux), dim=1)

        # Decoder
        decoder_out = self.decode(z_total)
        decoder_out_aux = self.decode_aux(z_aux)

        # Compute loss
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(decoder_out, x)
        recons_loss_aux = mse_loss(decoder_out_aux, x_aux)

        kl_loss = self.compute_kl(mu, logvar)
        kl_loss_aux = self.compute_kl(mu_aux, logvar_aux)

        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)

        self.log("Recons Loss Aux.", recons_loss_aux, prog_bar=True)
        self.log("Kl Loss Aux.", kl_loss_aux, prog_bar=True)

        total_loss = recons_loss + recons_loss_aux + self.alpha * (kl_loss + kl_loss_aux)
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:768"

    dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"


    aux_enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    aux_enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:256"

    aux_dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    aux_dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:256"

    vae = VAE(
        enc_block_config_str,
        dec_block_config_str,
        enc_channel_config_str,
        dec_channel_config_str,
        aux_enc_block_config_str,
        aux_dec_block_config_str,
        aux_enc_channel_config_str,
        aux_dec_channel_config_str,
    )

    sample = torch.randn(1, 3, 128, 128)
    out = vae.training_step(sample, 0)
    print(vae)
    print(out.shape)
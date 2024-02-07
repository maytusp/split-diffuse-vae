# FFHQ (Form1)
# python main/finetune_split_ddpm.py +dataset=ffhq_finetune/train \
#                      dataset.ddpm.data.root=\'/home/bias-team/Mo_Projects/DiffuseVAE/ffhq128\' \
#                      dataset.ddpm.data.name='ffhq' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=4 \
#                      dataset.ddpm.training.dummy_vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_vae_celebahq/vae_chq128_alpha=1.0_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1\' \
#                      dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq_finetune/\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.restore_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq_finetune/checkpoints/ddpmv2-ddpm_ffhq_v2-epoch=90-loss=0.0120.ckpt\' \
#                      dataset.ddpm.training.chkpt_prefix=\'ddpm_ffhq_v2\'

# # AFHQ Dog Train from scratch (Form1)
# python main/train_split_ddpm.py +dataset=afhq/train \
#                      dataset.ddpm.data.root=\'/home/bias-team/Mo_Projects/DiffuseVAE/afhq\' \
#                      dataset.ddpm.data.name='afhq_dog' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=6 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1\' \
#                      dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_dog\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'ffhq\' \
                     

# AFHQ Cat Train from scratch (Form1)
# python main/train_split_ddpm.py +dataset=afhq/train \
#                      dataset.ddpm.data.root=\'/home/bias-team/Mo_Projects/DiffuseVAE/afhq\' \
#                      dataset.ddpm.data.name='afhq_cat' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=6 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1\' \
#                      dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_cat\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'afhq\' \
                     

# CLEVR Train from scratch (Form1)
# python main/train_split_ddpm.py +dataset=clevr/train \
#                      dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/clevr/images/train' \
#                      dataset.ddpm.data.name='clevr' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=6 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-clevr-epoch=107-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1\' \
#                      dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_clevr\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'clevr\' \


# CARLA Train from scratch (Form1)
python main/train_split_ddpm.py +dataset=carla/train \
                     dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.ddpm.data.name='carla' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.model.dim=128 \
                     dataset.ddpm.model.dropout=0.1 \
                     dataset.ddpm.model.attn_resolutions=\'16,\' \
                     dataset.ddpm.model.n_residual=2 \
                     dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
                     dataset.ddpm.model.n_heads=8 \
                     dataset.ddpm.training.type='form1' \
                     dataset.ddpm.training.cfd_rate=0.0 \
                     dataset.ddpm.training.epochs=1000 \
                     dataset.ddpm.training.z_cond=False \
                     dataset.ddpm.training.batch_size=6 \
                     dataset.ddpm.training.device=\'gpu:0,1\' \
                     dataset.ddpm.training.workers=1 \
                     dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/checkpoints/vae-carla_day-epoch=1499-train_loss=0.0000.ckpt\' \
                     dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_carla_day\' \
                     dataset.ddpm.training.chkpt_prefix=\'carla\' \

# # FFHQ Train from scratch (Form1)
# python main/train_split_ddpm.py +dataset=ffhq/train \
#                      dataset.ddpm.data.root=\'/home/bias-team/Mo_Projects/DiffuseVAE/ffhq128\' \
#                      dataset.ddpm.data.name='ffhq' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=6 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1\' \
#                      dataset.ddpm.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'ffhq\' \
#                      dataset.ddpm.training.restore_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq/checkpoints/ddpmv2-ffhq-epoch=37-loss=0.0084.ckpt\' \

# # CelebAHQ-256 (Form1)
# python train_ddpm.py +dataset=celebahq/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/celeba_hq/\' \
#                      dataset.ddpm.data.name='celebahq' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.1 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,1,2,2,4,4\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.epochs=1000 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=8 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/vae-afhq256_10thJuly_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'tpu\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_chq256_rework_form1_10thJuly_sota_nheads=8_dropout=0.1/\' \
#                      dataset.ddpm.training.workers=1 \
#                      dataset.ddpm.training.chkpt_prefix=\'chq256_rework_form1_10thJuly_sota_nheads=8_dropout=0.1\'
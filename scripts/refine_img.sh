# Refine from Trained AFHQ Dog
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/split_vae_samp/' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=3 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_dog/checkpoints/ddpmv2-ffhq-epoch=280-loss=0.0048.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_ddpm_samp/\'
# Refine from Trained AFHQ Cat
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/split_vae_samp/' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=3 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_cat/checkpoints/ddpmv2-afhq-epoch=375-loss=0.0032.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_ddpm_samp/cat/\'

# # Refine from Trained AFHQ Dog (vary z global)
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/vary_z_global' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=20 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_dog/checkpoints/ddpmv2-ffhq-epoch=280-loss=0.0048.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/ddpm/vary_z_global\'

# # Refine from Trained AFHQ Dog (vary z local)
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/vary_z_local' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=20 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_dog/checkpoints/ddpmv2-ffhq-epoch=280-loss=0.0048.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/ddpm/vary_z_local\'

# # Refine from Trained AFHQ Cat (vary z global)
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/vary_z_global' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=20 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_cat/checkpoints/ddpmv2-afhq-epoch=375-loss=0.0032.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/ddpm/vary_z_global\'

# # Refine from Trained AFHQ Cat (vary z local)
# python main/eval/ddpm/generate_split_recons.py +dataset=afhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/vary_z_local' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=20 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_afhq_cat/checkpoints/ddpmv2-afhq-epoch=375-loss=0.0032.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/ddpm/vary_z_local\'

# # Refine image CLEVR (vary z global)
python main/eval/ddpm/generate_split_recons.py +dataset=clevr/test \
                        dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/vary_z_global' \
                        dataset.ddpm.data.name='ffhq' \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.hflip=False \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.1 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=20 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0,1\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=20 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=100 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_clevr/checkpoints/ddpmv2-clevr-epoch=19-loss=0.0020.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-clevr-epoch=107-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/ddpm/vary_z_global\'

# # Refine image CLEVR (vary z local)
python main/eval/ddpm/generate_split_recons.py +dataset=clevr/test \
                        dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/vary_z_local' \
                        dataset.ddpm.data.name='ffhq' \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.hflip=False \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.1 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=20 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0,1\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=20 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=100 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_clevr/checkpoints/ddpmv2-clevr-epoch=19-loss=0.0020.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-clevr-epoch=107-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/ddpm/vary_z_local\'

# Refine image CARLA (vary z global) (Train day, test day)
python main/eval/ddpm/generate_split_recons.py +dataset=carla/test \
                        dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/vary_z_global' \
                        dataset.ddpm.data.name='carla' \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.hflip=False \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.1 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=20 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0,1\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=20 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=100 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_carla_day/checkpoints/ddpmv2-carla-epoch=999-loss=0.0009.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/checkpoints/vae-carla_day-epoch=1499-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/ddpm/vary_z_global\'
# Refine image CARLA (vary z local) (Train day, test day)
python main/eval/ddpm/generate_split_recons.py +dataset=carla/test \
                        dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/vary_z_local' \
                        dataset.ddpm.data.name='carla' \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.hflip=False \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.1 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,4,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=20 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0,1\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=20 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=100 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_carla_day/checkpoints/ddpmv2-carla-epoch=999-loss=0.0009.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/checkpoints/vae-carla_day-epoch=1499-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/ddpm/vary_z_local\'



# Refine from Trained DDPM FFHQ
# python main/eval/ddpm/generate_split_recons.py +dataset=ffhq/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/vae_style_transfer/' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=20 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=3 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq/checkpoints/ddpmv2-ffhq-epoch=93-loss=0.0965.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_st/\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt\'


# Refine from Pretrained DDPM
# python main/eval/ddpm/refine_img_from_pt.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/vary_z_local_epoch216' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_ddpm_celebhq/pretrained_ddpm_chq128_form1_loss=0.0066.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=1 \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_vary_z_local_epoch216\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=20 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_vae_celebahq/vae_chq128_alpha=1.0_loss=0.0000.ckpt\'



# python main/eval/ddpm/refine_img.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/vary_z_global/003' \
#                         dataset.ddpm.data.name='ffhq' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_ddpm_celebhq/pretrained_ddpm_chq128_form1_loss=0.0066.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=1 \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_vary_global/003\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=20 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_vae_celebahq/vae_chq128_alpha=1.0_loss=0.0000.ckpt\'

                        


# dataset.ddpm.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/outputs/split_vae_normal/' \
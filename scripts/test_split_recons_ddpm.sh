python main/eval/ddpm/generate_recons.py +dataset=celebamaskhq128/test \
                        dataset.ddpm.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
                        dataset.ddpm.data.name='celebamaskhq' \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.data.hflip=False \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.dropout=0.1 \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.type='form2' \
                        dataset.ddpm.evaluation.resample_strategy='spaced' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddim' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=4 \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=100 \
                        dataset.ddpm.evaluation.n_steps=100 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celebahq128/ddpmv2-cmhq128_rework_form2_7thJuly_sota_nheads=8_dropout=0.1-epoch=999-loss=0.0032.ckpt\' \
                        dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt\' \
                        dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_trained_samp\'








# python main/eval/ddpm/generate_recons.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
#                         dataset.ddpm.data.name='celebamaskhq' \
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
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celebahq128/ddpmv2-cmhq128_rework_form2_7thJuly_sota_nheads=8_dropout=0.1-epoch=999-loss=0.0032.ckpt\' \
#                         dataset.ddpm.evaluation.type='form2' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=4 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_verify_reconstructions/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=100 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celebahq128/vae-cmhq128_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\'

# python main/eval/ddpm/generate_recons.py +dataset=cifar10/test \
#                         dataset.ddpm.data.root='/data1/kushagrap20/datasets/' \
#                         dataset.ddpm.data.name='cifar10' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.hflip=False \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.3 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cifar10/diffusevae_cifar10_rework_form1__7thJune_sota_nheads=8_dropout=0.3/checkpoints/ddpmv2-cifar10_rework_form1__7thJune_sota_nheads=8_dropout=0.3-epoch=2850-loss=0.0299.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=16 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_recons_superres/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=1000 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\'
# # # Split DDPM V1 For FFHQ
# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=16 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=10 \
#                         dataset.ddpm.evaluation.n_steps=50 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=32 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_ddpm_celebhq/pretrained_ddpm_chq128_form1_loss=0.0066.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_vae_celebahq/vae_chq128_alpha=1.0_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/celeba_ddpm_samp/\'



# # # dataset.vae.evaluation.expde_model_path=\'\'
# # # dataset.vae.evaluation.expde_model_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ffhq_latents/gmm_z/gmm_150.joblib\'
# # DDPM CelebHQ
# python main/eval/ddpm/sample_split_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=3 \
#                         dataset.ddpm.evaluation.sample_prefix='cpu' \
#                         dataset.ddpm.evaluation.device=\'cpu\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/pretrained_ddpm_celebhq/pretrained_ddpm_chq128_form1_loss=0.0066.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=0.8 \
#                         dataset.ddpm.evaluation.batch_size=1 \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_sampling/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=1 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq_v2/checkpoints/vae-ffhq128-epoch=99-train_loss=0.0000.ckpt\'


# # # DDPM CelebHQ
# python main/eval/ddpm/sample_split_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=3 \
#                         dataset.ddpm.evaluation.sample_prefix='cpu' \
#                         dataset.ddpm.evaluation.device=\'cpu\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=0.8 \
#                         dataset.ddpm.evaluation.batch_size=1 \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=50 \
#                         dataset.ddpm.evaluation.n_steps=50 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_miniimagenet/checkpoints/ddpmv2-ffhq-epoch=52-loss=0.0523.ckpt\' \
#                         dataset.vae.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_mini-imagenet/checkpoints/vae-mini-imagenet-epoch=147-train_loss=0.0000.ckpt\' \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/miniimagenet_ddpm_samp/\'



# python main/eval/ddpm/sample_split.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.0 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='cpu' \
#                         dataset.ddpm.evaluation.device=\'cpu\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/ddpm_ffhq128/checkpoints/ddpmv2-ffhq128_rework_form1__21stJune_sota_nheads=8_dropout=0.1-epoch=05-loss=0.0331.ckpt\' \
#                         dataset.ddpm.evaluation.type='uncond' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddim' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.batch_size=1 \
#                         dataset.ddpm.evaluation.save_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/outputs/ddpm_normal_sampling\' \
#                         dataset.ddpm.evaluation.n_samples=50 \
#                         dataset.ddpm.evaluation.n_steps=25 \
#                         dataset.ddpm.evaluation.workers=1


# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
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
#                         dataset.ddpm.evaluation.batch_size=16 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cmhq_confirm_form2/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=50 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celebahq128/vae-cmhq128_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                         dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/cmhq128_latents/gmm_z/gmm_100.joblib\'


# python main/eval/ddpm/sample.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_heads=1 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cifar10/ddpmv2-reproduce14Oct-epoch=2073-loss=0.0618.ckpt\' \
#                         dataset.ddpm.evaluation.type='uncond' \
#                         dataset.ddpm.evaluation.variance='fixedsmall' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_benchmark_speedvsquality/uncond/\' \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.workers=1 \


# python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.dim=160 \
#                         dataset.ddpm.model.dropout=0.3 \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.n_residual=3 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/VAEDM/ddpmv2-cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3_largermodel-epoch=2539-loss=0.0511.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/~/form1/2550/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
#                         dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/cifar10_latents/gmm_z/gmm_50.joblib\'


# python main/eval/ddpm/sample_cond.py +dataset=celeba64/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,2,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celeba64/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/checkpoints/ddpmv2-celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1-epoch=344-loss=0.0146.ckpt\' \
#                         dataset.ddpm.evaluation.type='form2' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_celeba64_benchmark_speedvsquality/form2/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/celeba64/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt\'
#                         dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/celeba64_latents/gmm_z/gmm_75.joblib\'


# python main/eval/ddpm/sample.py +dataset=celeba64/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,2,4\' \
#                         dataset.ddpm.model.n_heads=1 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celeba64/ddpmv2-celeba64_rework_uncond_28thJune_sota_nheads=1_dropout=0.1-epoch=499-loss=0.0133.ckpt\' \
#                         dataset.ddpm.evaluation.variance='fixedsmall' \
#                         dataset.ddpm.evaluation.type='uncond' \
#                         dataset.ddpm.evaluation.resample_strategy='spaced' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_celeba64_benchmark_speedvsquality/uncond_quad/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=100 \
#                         dataset.ddpm.evaluation.workers=1


# python main/eval/ddpm/sample_cond.py +dataset=celebahq/test \
#                         dataset.ddpm.data.ddpm_latent_path='/data1/kushagrap20/chq256_latents.pt' \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.dropout=0.1 \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,1,2,2,4,4\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=3 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_3' \
#                         dataset.ddpm.evaluation.device=\'gpu:0,1,2,3\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/celebahq256/ddpmv2-celebahq256_rework_form1__Jul12th_sota_nheads=8_dropout=0.1-epoch=787-loss=0.0361.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=0.8 \
#                         dataset.ddpm.evaluation.batch_size=8 \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/paper_samples_chq256_sharedlatents/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=256 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/vae-celebahq256_alpha=1.0_Jan31-epoch=499-train_loss=0.0000.ckpt\'
#                         dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/celebahq_latents/gmm_z/gmm_100.joblib\'

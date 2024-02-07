# CARLA training
# python main/train_ae.py +dataset=carla/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
#                      dataset.vae.data.name='carla' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=16 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_carla_day/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'carla_day\' \
#                      dataset.vae.training.alpha=1.0 \

# CARLA Seed3
python main/train_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_carla_day3/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=3

# CARLA Seed5
python main/train_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_carla_day5/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=5
# CARLA Seed6
python main/train_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_carla_day6/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=6

# CARLA Seed10
python main/train_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_carla_day10/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=10

# # CelebAMaskHQ training
# python main/train_ae.py +dataset=celebamaskhq128/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
#                      dataset.vae.data.name='celebamaskhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=42 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_cmhq128_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'cmhq128_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# FFHQ 128 training
# python main/train_ae.py +dataset=ffhq/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/ffhq128' \
#                      dataset.vae.data.name='ffhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=16 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vanilla_vae_ffhq/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'ffhq128\' \
#                      dataset.vae.training.alpha=1.0

# # AFHQv2 training
# python main/train_ae.py +dataset=afhq256/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/afhq_v2/' \
#                      dataset.vae.data.name='afhq' \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_afhq256_10thJuly_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq256_10thJuly_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0


# # CelebA training
# python main/train_ae.py +dataset=celeba64/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.vae.data.name='celeba' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0
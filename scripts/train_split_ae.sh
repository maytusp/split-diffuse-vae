
# #  AFHQ Dog training
# python main/train_split_ae.py +dataset=afhq/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/afhq' \
#                      dataset.vae.data.name='afhq_dog' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=3000 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq\' \
#                      dataset.vae.training.alpha=1.0 \
# #  AFHQ Cat training
# python main/train_split_ae.py +dataset=afhq/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/afhq' \
#                      dataset.vae.data.name='afhq_cat' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=3000 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq\' \
#                      dataset.vae.training.alpha=1.0 \
# # CLEVR training
# python main/train_split_ae.py +dataset=clevr/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/clevr/images/train' \
#                      dataset.vae.data.name='clevr' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=16 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'mini-imagenet\' \
#                      dataset.vae.training.alpha=1.0 \

# CARLA training
# python main/train_split_ae.py +dataset=carla/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
#                      dataset.vae.data.name='carla' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=16 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'carla_day\' \
#                      dataset.vae.training.alpha=1.0 \

# CARLA training seed 4
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day4/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=4

# CARLA training seed 5
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day5/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=5

# CARLA training seed 6
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day6/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=6

# CARLA training seed 7
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day7/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=7

# CARLA training seed 9
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day9/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=9

# CARLA training seed 10
python main/train_split_ae.py +dataset=carla/train \
                     dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/carla/day_train' \
                     dataset.vae.data.name='carla' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=16 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=1500 \
                     dataset.vae.training.device=\'gpu:1\' \
                     dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day10/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'carla_day\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.vae.training.seed=10

# FFHQ 128 training
# python main/train_split_ae.py +dataset=ffhq/train \
#                      dataset.vae.data.root='/home/bias-team/Mo_Projects/DiffuseVAE/ffhq128' \
#                      dataset.vae.data.name='ffhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=16 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1\' \
#                      dataset.vae.training.results_dir=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq_v2/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'ffhq128\' \
#                      dataset.vae.training.alpha=1.0 \
#                      dataset.vae.training.restore_path=\'/home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq_v2/checkpoints/vae-ffhq128-epoch=138-train_loss=0.0000.ckpt\'

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
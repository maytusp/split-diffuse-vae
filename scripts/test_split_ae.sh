# # Normal sample
# python main/test_split.py sample --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 200 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/split_vae_samp/ \
#                                 --write-mode image \
#                                 1024 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt \
# Reconstruct AFHQ Dog
# python main/test_split.py reconstruct --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 200 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/split_vae_samp/ \
#                                 --write-mode image \
#                                 --dataset afhq_dog \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq
# # # Reconstruct AFHQ Cat
# python main/test_split.py reconstruct --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 200 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/split_vae_samp/ \
#                                 --write-mode image \
#                                 --dataset afhq_cat \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq

# # # Reconstruct CLEVR
# python main/test_split.py reconstruct --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 200 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/split_vae_samp/ \
#                                 --write-mode image \
#                                 --dataset clevr \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-mini-imagenet-epoch=92.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/clevr/images/val


# # Vary z_global AFHQ Dog
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/vary_z_global \
#                                 --write-mode image \
#                                 --vary z_global \
#                                 --dataset afhq_dog \
#                                 768 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq/val

# # Vary z_local AFHQ Dog
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_dog/vary_z_local \
#                                 --write-mode image \
#                                 --vary z_local \
#                                 --dataset afhq_dog \
#                                 256 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_dog/checkpoints/vae-afhq-epoch=934-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq/val


# # Vary z_global AFHQ Cat
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/vary_z_global \
#                                 --write-mode image \
#                                 --vary z_global \
#                                 --dataset afhq_cat \
#                                 768 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq/val

# # Vary z_local AFHQ Cat
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/afhq_cat/vary_z_local \
#                                 --write-mode image \
#                                 --vary z_local \
#                                 --dataset afhq_cat \
#                                 256 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_afhq_cat/checkpoints/vae-afhq-epoch=1083-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/afhq/val



# # # Vary z_global CLEVR
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/vary_z_global \
#                                 --write-mode image \
#                                 --vary z_global \
#                                 --dataset clevr \
#                                 768 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-clevr-epoch=107-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/clevr/images/val
# # # Vary z_local CLEVR
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/clevr/vary_z_local \
#                                 --write-mode image \
#                                 --vary z_local \
#                                 --dataset clevr \
#                                 256 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_clevr/checkpoints/vae-clevr-epoch=107-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/clevr/images/val


# # Vary z_global CARLA
python main/test_split.py sample-vary-z --device cpu \
                                --image-size 128 \
                                --num-samples 20 \
                                --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/vary_z_global \
                                --write-mode image \
                                --vary z_global \
                                --dataset carla \
                                768 \
                                /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/checkpoints/vae-carla_day-epoch=1499-train_loss=0.0000.ckpt \
                                /home/bias-team/Mo_Projects/DiffuseVAE/carla/day_val/

# # Vary z_local CARLA
python main/test_split.py sample-vary-z --device cpu \
                                --image-size 128 \
                                --num-samples 20 \
                                --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/carla/day_day/vary_z_local \
                                --write-mode image \
                                --vary z_local \
                                --dataset carla \
                                256 \
                                /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_carla_day/checkpoints/vae-carla_day-epoch=1499-train_loss=0.0000.ckpt \
                                /home/bias-team/Mo_Projects/DiffuseVAE/carla/day_val/

# # Normal sample
# python main/test_split.py sample --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 50 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/split_vae_samp/ \
#                                 --write-mode image \
#                                 1024 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt \

# # # # vary z_local
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 3 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/examples/ \
#                                 --write-mode image \
#                                 --vary z_local \
#                                 256 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/examples_img/

# # vary z_global
# python main/test_split.py sample-vary-z --device cpu \
#                                 --image-size 128 \
#                                 --num-samples 20 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/vary_z_global/ \
#                                 --write-mode image \
#                                 --vary z_global \
#                                 768 \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/ffhq_toy/

# style transfer
# python main/test_split.py style-transfer --device cpu \
#                                 --image-size 128 \
#                                 --save-path /home/bias-team/Mo_Projects/DiffuseVAE/outputs/vae_style_transfer/ \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/logs/vae_ffhq/checkpoints/vae-ffhq128-epoch=216-train_loss=0.0000.ckpt \
#                                 /home/bias-team/Mo_Projects/DiffuseVAE/outputs/ori_for_st/


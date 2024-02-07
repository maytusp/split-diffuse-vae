import random
from matplotlib import pyplot as plt
import albumentations as A
import os
from PIL import Image
import numpy as np

source_dir = "/home/bias-team/Mo_Projects/DiffuseVAE/ood_experiment/carla2_source"
train_dir = "/home/bias-team/Mo_Projects/DiffuseVAE/ood_experiment/carla2_ood_hard_original_size/train"
test_dir = "/home/bias-team/Mo_Projects/DiffuseVAE/ood_experiment/carla2_ood_hard_original_size/test"


def visualize(image):
    plt.figure(figsize=(8, 5))
    plt.axis('off')
    plt.imshow(image)

def read_img(img_path):
    image = Image.open(img_path)
    image = np.array(image)
    if image.shape[2] > 3:
        image = image[:,:,:3]
    return image

def save_img(img_array, img_path):
    image = Image.fromarray(img_array)
    image.save(img_path)

def add_rain(image):
    brightness = float(np.random.uniform(0.6,0.9,1))
    blurness = int(np.random.uniform(6,8,1))
    width = int(np.random.randint(1,3,1))
    transform = A.Compose(
        [A.RandomRain(brightness_coefficient=brightness, drop_width=width, blur_value=blurness, p=1)],
    )
    transformed = transform(image=image)
    return transformed['image']

def add_fog(image):
    alpha = float(np.random.uniform(0.1,0.15,1))
    transform = A.Compose(
        [A.RandomFog(fog_coef_lower=0.8, fog_coef_upper=0.8, alpha_coef=alpha, p=1)],
    )
    transformed = transform(image=image)
    return transformed['image']

def add_flare(image):
    r, g, b = np.random.randint(250,255,3)
    r = int(r)
    g = int(g)
    b = int(b)
    transform = A.Compose(
        [A.RandomSunFlare(flare_roi=(0.3, 0, 0.7, 0.1), angle_lower=0.7, p=1, src_color=(r, g, b))],
    )
    transformed = transform(image=image)
    return transformed['image']

def add_snow(image):
    brightness_coeff = int(np.random.randint(4,7,1))
    transform = A.Compose(
        [A.RandomSnow(brightness_coeff=3, snow_point_lower=0.1, snow_point_upper=0.3, p=1)],
    )
    transformed = transform(image=image)
    return transformed['image']

def add_shadow(image):
    transform = A.Compose(
        [A.RandomShadow(num_shadows_lower=10, num_shadows_upper=10, shadow_dimension=10, shadow_roi=(0, 0.5, 1, 1), p=1)],
    )
    transformed = transform(image=image)
    return transformed['image']

def augment(source_dir, train_dir, test_dir):
    cls_list = os.listdir(source_dir)

    for cls in cls_list:
        subclass_source_dir = os.path.join(source_dir, cls)
        subclass_file_list = os.listdir(subclass_source_dir)

        subclass_train_dir = os.path.join(train_dir, cls)
        subclass_test_dir = os.path.join(test_dir, cls)

        if not(os.path.exists(subclass_train_dir)):
            os.makedirs(subclass_train_dir)

        if not(os.path.exists(subclass_test_dir)):
            os.makedirs(subclass_test_dir)
        
        
        for fn in subclass_file_list:
            img_path = os.path.join(subclass_source_dir, fn)
            ori_img = read_img(img_path)

            save_img(ori_img, os.path.join(subclass_train_dir, f"{fn}"))

            for iter in range(10):
                # rain_img = add_rain(ori_img)
                # fog_img = add_fog(ori_img)
                flare_img = add_flare(ori_img)
                # snow_img = add_snow(ori_img)
                # shadow_img = add_shadow(ori_img)
                
                # save_img(rain_img, os.path.join(subclass_test_dir, f"rain_{iter}_{fn}"))
                # save_img(fog_img, os.path.join(subclass_test_dir, f"fog_{iter}_{fn}"))
                save_img(flare_img, os.path.join(subclass_test_dir, f"flare_{iter}_{fn}"))
                # save_img(snow_img, os.path.join(subclass_test_dir, f"snow_{iter}_{fn}"))
                # save_img(shadow_img, os.path.join(subclass_test_dir, f"shadow_{iter}_{fn}"))


augment(source_dir, train_dir, test_dir)
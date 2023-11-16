import os
import cv2
import tqdm
import shutil
import random


def main():
    """
    入口文件
    :return: None
    """
    root_dir = r"emotic"
    save_dir = r"train"
    make_new_dir(root_dir, "train", save_dir)  # 制作训练集

    root_dir = r"emotic"
    save_dir = r"test"
    make_new_dir(root_dir, "test", save_dir)  # 制作测试集

    # 随机移动10%数量的图像到val文件夹中
    input_folder = r"train"
    save_folder = r"val"
    move_2_new_dir(input_folder, save_folder)


def make_new_dir(root_dir, tag, save_dir):
    """
    将文件移动到目标目录中并更改文件夹内jpg格式文件为文件名
    :param root_dir: 原目录
    :param tag: 文件夹名，test或者是train
    :param save_dir: 目标目录
    :return: None
    """
    img_root = os.path.join(root_dir, tag)
    img_dir_list = os.listdir(img_root)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print(img_dir_list)
    for img_dir in tqdm.tqdm(img_dir_list):
        img_folder = os.path.join(img_root, img_dir)
        img_list = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
        name_first = img_dir[:2].upper()  # 表情名称前两位
        for index, filename in enumerate(img_list):
            img = cv2.imread(os.path.join(img_folder, filename), -1)
            save_name = name_first+"{0:0>5}.jpg".format(index)  # 更改图片名为表情名+五位编号的形式
            cv2.imwrite(os.path.join(save_dir, save_name), img)


def move_2_new_dir(input_folder, save_folder):
    """
    随机提取10%的训练集数据到val文件夹中
    :param input_folder:
    :param save_folder:
    :return:
    """
    filenames = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    random.shuffle(filenames)
    num_val = int(0.1 * len(filenames))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    for index, filename in enumerate(filenames):
        src = os.path.join(input_folder, filename)
        dst = os.path.join(save_folder, filename)
        shutil.move(src, dst)
        if index == num_val:
            break


main()

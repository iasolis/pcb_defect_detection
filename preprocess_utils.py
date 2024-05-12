import xml.etree.ElementTree as ET
from os import makedirs, listdir
import random

from PIL import Image


def make_train_data_structure(dirs: list[str]) -> None:
    """
    Функция создания структуры директорий для обучающих данных.
    Parameters
    ----------
    dirs: list[str]
        Список папок, которые будут осздаваться.
    """
    for _dir in dirs:
        makedirs(_dir, exist_ok=True)


def divide_train_val_test(img_dir, anno_dir, subdirs, train_size: 0.7) -> tuple:
    img_train_paths, img_val_paths,  = [], []
    anno_train_paths, anno_val_paths,  = [], []
    for subdir in subdirs:
        img_dir_path = img_dir + subdir
        anno_dir_path = anno_dir + subdir

        img_paths = [img_dir_path + f for f in listdir(img_dir_path)]
        anno_paths = [anno_dir_path + f for f in listdir(anno_dir_path)]

        number_of_train = int(len(img_paths) * train_size)
        for _ in range(number_of_train):
            random_idx = random.randint(0, len(img_paths) - 1)
            img_train_path = img_paths.pop(random_idx)
            img_train_paths.append(img_train_path)

        anno_val_paths.extend(img_paths)
        anno_val_paths.extend(anno_paths)
    return img_train_paths, img_val_paths, anno_train_paths, anno_val_paths


def resize_img(img_path: str, target_size: int = 2240) -> Image.Image:
    """
    Функция для изменения разрешения изображения
    Parameters
    ----------
    img_path: str
        Путь до изображения, разрешение которого будет изменятся
    target_size: int, int
        Конечный размер изображения
    Returns
    -------
    Image.Image
        Изображение в виде объекта класса PIL.Image.Image
    """
    img = Image.open(img_path)
    img_size = (target_size, target_size)
    resized_img = img.resize(img_size)
    return resized_img

def take_anno_params(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = root.find('size').find('width').text
    height = root.find('size').find('height').text
    label = root.find('object').find('name').text

    img_size = (int(width), int(height))

    bndboxs = []
    for obj in root.iter('object'):
        for box in obj.iter('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            bndboxs.append((xmin, ymin, xmax, ymax))
    return label, bndboxs, img_size


def resize_anno_params(bndboxs: list[tuple], old_img_size: tuple, target_size: int = 2240) -> tuple[list, tuple]:
    """
    Функция для изменения параметров избражения в аннотации, а именно: разрешения изображения и координат ограничивающей рамки
    Parameters
    ----------
    bndboxs: list
        Координаты ограничивающей рамки (xmin, ymin, xmax, ymax)
    old_img_size: tuple
        Разрмер изначального изображения (width, height)
    target_size: int
        Конечный размер изображения
    Returns
    -------
    tuple[list, tuple]
        Два кортежа, первый содержит измененные координаты изображения, а второй размеры
    """
    resized_size = (target_size, target_size)
    resized_bndboxs = []
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        old_width, old_height = old_img_size

        xmin = float(xmin * target_size / old_width)
        ymin = float(ymin * target_size / old_height)
        xmax = float(xmax * target_size / old_width)
        ymax = float(ymax * target_size / old_height)

        resized_bndboxs.append((xmin, ymin, xmax, ymax))
    return resized_bndboxs, resized_size











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


def divide_train_val(img_dir: str, anno_dir: str, subdirs: list, train_size: float = 0.8) -> tuple:
    """
    Функция разделения на тренировочную и валидационную выборки для изображений и анотаций
    Parameters
    ----------
    img_dir:
        Путь до папки с оригинальными изображениями
    anno_dir:
        Путь до папки с оригинальными аннотациями
    subdirs:
        Имена подпапок, соответствуют классам изображений
    train_size:
        Относительный размер тренировочной выборки, например 0.8 означает,
        что тренировочная выборка будет составлять 80% данных от оригинальной

    Returns
    -------
    tuple
        Возвращает кортеж списков, которые содержат пути до оригинальных изображений и аннотаций, разделенные на tarin/val
    """
    img_train_paths, img_val_paths, = [], []
    anno_train_paths, anno_val_paths, = [], []
    img_test_paths, anno_test_paths = [], []
    for subdir in subdirs:
        img_dir_path = img_dir + subdir
        anno_dir_path = anno_dir + subdir

        img_paths = [img_dir_path + f for f in listdir(img_dir_path)]
        anno_paths = [anno_dir_path + f for f in listdir(anno_dir_path)]

        number_of_train = int(len(img_paths) * train_size)
        for _ in range(number_of_train):
            random_idx = random.randint(0, len(img_paths) - 1)

            img_train_path = img_paths.pop(random_idx)
            anno_train_path = anno_paths.pop(random_idx)

            img_train_paths.append(img_train_path)
            anno_train_paths.append(anno_train_path)

        img_val_paths.extend(img_paths)
        anno_val_paths.extend(anno_paths)

        img_test_paths.extend(img_val_paths[:5])
        anno_test_paths.extend(anno_val_paths[:5])

    return img_train_paths, img_val_paths, img_test_paths, anno_train_paths, anno_val_paths, anno_test_paths


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


def take_anno_params(xml_path: str) -> tuple:
    """
    Функция для парсинга ключевых параметров аннотиции:
    лэйбла изображения, координат окружающего треугольника и размера изображения
    Parameters
    ----------
    xml_path:
        Путь до аннотации
    Returns
    -------
    tuple
        Кортеж содержащий лэйбл изображения, кортеж координат окружающего треугольника и  кортеж размеров изображения
    """
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


def resize_anno_params(bndboxs: list[tuple], old_img_size: tuple, target_size: int = 640) -> tuple[list, tuple]:
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
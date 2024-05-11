from yolo_config import ORIGINAL_DATA_IMG_DIR, ORIGINAL_DATA_ANNO_DIR, ORIGINAL_DATA_SUBDIRS, \
                        DATA_DIR, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR, TARGET_SIZE
import xml.etree.ElementTree as ET
from os import makedirs, listdir
from os.path import exists, basename, splitext
import random

from PIL import Image
import yaml


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


def take_filepaths(data_dir, subdirs) -> list:
    filepaths = []
    for subdir in subdirs:
        dir_path = data_dir + subdir
        filepaths.extend([data_dir + subdir + f for f in listdir(dir_path)])
    return filepaths


def divide_train_val(img_filepaths: list[str], anno_filepaths: list[str], train_size: float = 0.7) -> tuple:
    """
    Разделение изображений на тренировочный, валидационный  и тренировочный датасеты.
    Parameters
    ----------
    img_filepaths:
        Путь до изображения.
    anno_filepaths:
        Путь до аннотации изображения.
    train_size:
        Размер тренировочного датасета в относительных еденицах, например 0.7 означает, что 70%
        изображений и аннотаций будут тренировочными.
    Returns
    -------
    tuple:
        Кортеж путей до всех файлов, которые делились на выборки:
         img_train_paths, img_val_paths, anno_train_paths, anno_val_paths
    """
    imgs = img_filepaths.copy()
    annos = anno_filepaths.copy()
    number_of_train = int(len(imgs) * train_size)

    img_train_paths, img_val_paths, anno_train_paths, anno_val_paths = [], [], [], []
    for _ in range(number_of_train):
        random_idx = random.randint(0, len(imgs) - 1)

        img = imgs.pop(random_idx)
        anno = annos.pop(random_idx)

        img_train_paths.append(img)
        anno_train_paths.append(anno)

    img_val_paths.extend(imgs)
    anno_val_paths.extend(annos)
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
    return bndboxs, img_size, label


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


def save_anno_in_yolo_format(label: str, bndboxs: list, path_to_save: str, img_size: tuple) -> None:
    """
    Функция сохранения label.txt файла для yolo модели
    Parameters
    ----------
    label:
        Название дефекта (лэйбл)
    bndboxs:
        Координаты ограничивающей рамки (xmin, ymin, xmax, ymax)
    path_to_save:
        Путь сохранения созданных .txt файлов
    img_size:
        Размер изображения, лэйблы которого сохраняются
    """
    labels = {'spurious_copper': 0,
              'mouse_bite': 1,
              'open_circuit': 2,
              'missing_hole': 3,
              'spur': 4,
              'short': 5}
    width, height = img_size
    with open(path_to_save, "w") as file:
        for bndbox in bndboxs:
            xmin, ymin, xmax, ymax = bndbox
            lbl = labels[label]
            string_to_save = f'{lbl} {(xmin+xmax)/2/width} {(ymin+ymax)/2/height} {(xmax-xmin)/width} {(ymax-ymin)/height} \n'
            file.write(string_to_save)


def save_yaml_yolo():
    """
    Функция сохранения .yaml файла для модели yolo.
    """
    data = {'path': 'data/',
            'train': 'images/train/',
            'val': 'images/val/',
            'nc': 6,
            'names': {0: 'spurious_copper',
                      1: 'mouse_bite',
                      2: 'open_circuit',
                      3: 'missing_hole',
                      4: 'spur',
                      5: 'short'
                      }
            }
    with open('datasets/data/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_train_val_files(orig_img_train_paths: str, orig_img_val_paths: str, orig_anno_train_paths: str, orig_anno_val_paths: str,
                         train_img_dir: str, val_img_dir: str, train_anno_dir: str, val_anno_dir: str,
                         target_img_size: int) -> None:
    """
    Функция сохранения всех получившихся файлов.
    Parameters
    ----------
    orig_img_train_paths:
        Путь к оригинальному изображению, которое было отнесено к тренировочным данным
    orig_img_val_paths:
        Путь к оригинальному изображению, которое было отнесено к валидационным данным
    orig_anno_train_paths:
        Путь к оригинальной аннотации изображения, которое было отнесено к тренировочным данным
    orig_anno_val_paths:
        Путь к оригинальной аннотации изображения, которое было отнесено к валидационным данным
    train_img_dir:
        Путь до папки в которую будут сохраняться тренировочные изображения
    val_img_dir:
        Путь до папки в которую будут сохраняться валидационным изображения
    train_anno_dir:
            Путь до папки в которую будут сохраняться тренировочные аннотации
    val_anno_dir:
                    Путь до папки в которую будут сохраняться валидационным аннотации
    target_img_size:
        Размер изображения до которого будет производиться изменение размера
    """
    for img_path, anno_path in zip(orig_img_train_paths, orig_anno_train_paths):
        img = resize_img(img_path, target_img_size)
        img_name = basename(img_path)
        img_save_path = train_img_dir + img_name
        img.save(img_save_path)

        bndbox, img_size, label = take_anno_params(anno_path)
        bndbox, img_size = resize_anno_params(bndbox, img_size, target_img_size)
        anno_name = splitext(basename(anno_path))[0]
        anno_save_path = train_anno_dir + anno_name + '.txt'

        save_anno_in_yolo_format(label, bndbox, anno_save_path, img_size)

    for img_path, anno_path in zip(orig_img_val_paths, orig_anno_val_paths):
        img = resize_img(img_path, target_img_size)
        img_name = basename(img_path)
        img_save_path = val_img_dir + img_name
        img.save(img_save_path)

        bndbox, img_size, label = take_anno_params(anno_path)
        bndbox, img_size = resize_anno_params(bndbox, img_size, target_img_size)
        anno_name = splitext(basename(anno_path))[0]
        anno_save_path = val_anno_dir + anno_name + '.txt'
        save_anno_in_yolo_format(label, bndbox, anno_save_path, img_size)

def main():
    if not exists(DATA_DIR):
        train_data_dirs = [DATA_DIR, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR]
        make_train_data_structure(train_data_dirs)

    img_dir = ORIGINAL_DATA_IMG_DIR
    anno_dir = ORIGINAL_DATA_ANNO_DIR
    subdirs = ORIGINAL_DATA_SUBDIRS

    train_img_dir = TRAIN_IMG_DIR
    val_img_dir = VAL_IMG_DIR
    train_anno_dir = TRAIN_ANNO_DIR
    val_anno_dir = VAL_ANNO_DIR

    target_img_size = TARGET_SIZE

    img_filepaths = take_filepaths(img_dir, subdirs)
    anno_filepaths = take_filepaths(anno_dir, subdirs)

    img_train_paths, img_val_paths, anno_train_paths, anno_val_paths = divide_train_val(img_filepaths,
                                                                                        anno_filepaths, train_size=0.7)

    save_train_val_files(img_train_paths, img_val_paths, anno_train_paths, anno_val_paths,
                         train_img_dir, val_img_dir, train_anno_dir, val_anno_dir, target_img_size)
    save_yaml_yolo()


if __name__ == '__main__':
    main()

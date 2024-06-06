from config import ORIGINAL_DATA_IMG_DIR, ORIGINAL_DATA_ANNO_DIR, ORIGINAL_DATA_SUBDIRS, \
    DATA_DIR, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR, TEST_IMG_DIR, TEST_ANNO_DIR, TARGET_SIZE
from augmentation import augment_orig_data
from image_processing import symmetrize_img

from os.path import exists, basename, splitext
from os import makedirs, listdir
import xml.etree.ElementTree as ET
import random
import argparse

from tqdm import tqdm
import numpy as np
import yaml
import cv2


def save_yolo_yaml():
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
    with open('yolov8/datasets/data/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


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


def divide_train_val_test(img_dir: str, anno_dir: str, subdirs: list, train_size: float = 0.7) -> tuple:
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
    names_of_labels = {'spurious_copper': 0,
                       'mouse_bite': 1,
                       'open_circuit': 2,
                       'missing_hole': 3,
                       'spur': 4,
                       'short': 5}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = root.find('size').find('width').text
    height = root.find('size').find('height').text

    img_size = (int(width), int(height))
    labels = []
    bndboxs = []
    for obj in root.iter('object'):
        name_label = root.find('object').find('name').text
        labels.append(names_of_labels[name_label])
        for box in obj.iter('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            bndboxs.append((xmin, ymin, xmax, ymax))
    return labels, bndboxs, img_size


def load_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    return img


def save_img(img_path: str, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)


def save_label(anno_path: str, bboxs: list, labels: list) -> None:
    with open(anno_path, "w") as file:
        for bbox, label in zip(bboxs, labels):
            x, y, w, h = bbox
            string_to_save = f'{label} {x} {y} {w} {h}\n'
            file.write(string_to_save)


def resize_bbox_to_target_size(bbox, old_img_size, target_size):
    xmin, ymin, xmax, ymax = bbox
    old_width, old_height = old_img_size

    xmin = float(xmin * target_size / old_width)
    ymin = float(ymin * target_size / old_height)
    xmax = float(xmax * target_size / old_width)
    ymax = float(ymax * target_size / old_height)

    resized_box = [xmin, ymin, xmax, ymax]
    return resized_box


def reformat_bboxs_to_yolo_format(bboxs: list, old_img_size: tuple, target_size: int = 640) -> list:
    """
    Перевод ограничивающиех прямоугольников из формата voc в yolo
    Parameters
    ----------
    bboxs:
        Ограничивающие прямоугольники
    old_img_size:
        Размер оригинального изображения
    target_size:
        Целевой размер изображения
    Returns
    -------
    list
         Переформатированные ограничивающие прямоугольники
    """
    yolo_bboxs = []
    for bbox in bboxs:
        xmin, ymin, xmax, ymax = resize_bbox_to_target_size(bbox, old_img_size, target_size)
        x = (xmin + xmax) / 2 / target_size
        y = (ymin + ymax) / 2 / target_size
        w = (xmax - xmin) / target_size
        h = (ymax - ymin) / target_size
        yolo_bboxs.append([x, y, w, h])
    return yolo_bboxs


def save_yolo_files(orig_img_paths: str, orig_anno_paths: str, save_img_dir: str, save_label_dir: str,
                    target_img_size: int, aug_flag:bool=False) -> None:
    """
    Функция сохранения всех получившихся файлов.
    Parameters
    ----------
    orig_img_paths:
        Путь к оригинальному изображению
    orig_anno_paths:
        Путь к оригинальной аннотации изображения
    save_img_dir:
        Путь до папки в которую будут сохраняться изображения
    save_label_dir:
            Путь до папки в которую будут сохраняться аннотации
    target_img_size:
        Размер изображения до которого будет производиться изменение размера
    """
    for img_path, anno_path in tqdm(zip(orig_img_paths, orig_anno_paths)):
        img_name = basename(img_path)
        img_save_path = save_img_dir + img_name

        label_name = splitext(basename(anno_path))[0]
        label_save_path = save_label_dir + label_name + '.txt'

        img = load_img(img_path)
        resized_img = symmetrize_img(img, target_img_size)
        save_img(img_save_path, resized_img)

        names, bboxs, img_size = take_anno_params(anno_path)
        yolo_bboxs = reformat_bboxs_to_yolo_format(bboxs, img_size, target_img_size)
        save_label(label_save_path, yolo_bboxs, names)

        if aug_flag:
            aug_imgs, aug_bboxes, aug_names = augment_orig_data(img, yolo_bboxs, names)
            for aug_img, aug_bbox, aug_name in zip(aug_imgs, aug_bboxes, aug_names):
                aug_img_save_path = save_img_dir + aug_name + img_name
                aug_label_save_path = save_label_dir + aug_name + label_name + '.txt'

                save_img(aug_img_save_path, aug_img)
                save_label(aug_label_save_path, aug_bbox, names)


def main():
    if not exists(DATA_DIR):
        train_data_dirs = [DATA_DIR, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR, TEST_IMG_DIR,
                           TEST_ANNO_DIR]
        make_train_data_structure(train_data_dirs)

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_flag', required=False, default=False)
    parser.add_argument('--image_processing_flag', required=False, default=False)
    opt = parser.parse_args()

    img_dir = ORIGINAL_DATA_IMG_DIR
    anno_dir = ORIGINAL_DATA_ANNO_DIR
    subdirs = ORIGINAL_DATA_SUBDIRS

    train_img_dir = TRAIN_IMG_DIR
    train_anno_dir = TRAIN_ANNO_DIR

    train_val_test_paths = divide_train_val_test(img_dir, anno_dir, subdirs, train_size=0.8)
    img_train_paths, img_val_paths, img_test_paths, anno_train_paths, anno_val_paths, anno_test_paths = train_val_test_paths

    save_yolo_files(img_train_paths, anno_train_paths, train_img_dir, train_anno_dir, TARGET_SIZE, opt.aug_flag)
    save_yolo_files(img_val_paths, anno_val_paths, VAL_IMG_DIR, VAL_ANNO_DIR, TARGET_SIZE, opt.aug_flag)
    save_yolo_files(img_test_paths, anno_test_paths, TEST_IMG_DIR, TEST_ANNO_DIR, TARGET_SIZE, opt.aug_flag)
    save_yolo_yaml()


if __name__ == '__main__':
    main()

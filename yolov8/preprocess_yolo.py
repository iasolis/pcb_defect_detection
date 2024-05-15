from config_yolo import ORIGINAL_DATA_IMG_DIR, ORIGINAL_DATA_ANNO_DIR, ORIGINAL_DATA_SUBDIRS, \
    DATA_DIR, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR, TEST_IMG_DIR, TEST_ANNO_DIR, TARGET_SIZE
from preprocess_yolo_utils import  make_train_data_structure, divide_train_val, resize_img, resize_anno_params, take_anno_params
from os.path import exists, basename, splitext

import yaml


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
            string_to_save = f'{lbl} {(xmin + xmax) / 2 / width} {(ymin + ymax) / 2 / height} {(xmax - xmin) / width} {(ymax - ymin) / height} \n'
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


def save_yolo_train_val_files(orig_img_train_paths: str, orig_img_val_paths: str, orig_img_test_paths: str,
                              orig_anno_train_paths: str, orig_anno_val_paths: str, orig_anno_test_paths: str,
                              train_img_dir: str, val_img_dir: str, test_img_dir: str,
                              train_anno_dir: str, val_anno_dir: str, test_anno_dir: str,
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

        label, bndbox, img_size = take_anno_params(anno_path)
        bndbox, img_size = resize_anno_params(bndbox, img_size, target_img_size)
        anno_name = splitext(basename(anno_path))[0]
        anno_save_path = train_anno_dir + anno_name + '.txt'

        save_anno_in_yolo_format(label, bndbox, anno_save_path, img_size)

    for img_path, anno_path in zip(orig_img_val_paths, orig_anno_val_paths):
        img = resize_img(img_path, target_img_size)
        img_name = basename(img_path)
        img_save_path = val_img_dir + img_name
        img.save(img_save_path)

        label, bndbox, img_size = take_anno_params(anno_path)
        bndbox, img_size = resize_anno_params(bndbox, img_size, target_img_size)
        anno_name = splitext(basename(anno_path))[0]
        anno_save_path = val_anno_dir + anno_name + '.txt'
        save_anno_in_yolo_format(label, bndbox, anno_save_path, img_size)

    for img_path, anno_path in zip(orig_img_test_paths, orig_anno_test_paths):
        img = resize_img(img_path, target_img_size)
        img_name = basename(img_path)
        img_save_path = test_img_dir + img_name
        img.save(img_save_path)

        label, bndbox, img_size = take_anno_params(anno_path)
        bndbox, img_size = resize_anno_params(bndbox, img_size, target_img_size)
        anno_name = splitext(basename(anno_path))[0]
        anno_save_path = test_anno_dir + anno_name + '.txt'
        save_anno_in_yolo_format(label, bndbox, anno_save_path, img_size)


def main():
    if not exists(DATA_DIR):
        train_data_dirs = [DATA_DIR, TRAIN_IMG_DIR,TRAIN_ANNO_DIR, VAL_IMG_DIR, VAL_ANNO_DIR, TEST_IMG_DIR, TEST_ANNO_DIR]
        make_train_data_structure(train_data_dirs)

    img_dir = ORIGINAL_DATA_IMG_DIR
    anno_dir = ORIGINAL_DATA_ANNO_DIR
    subdirs = ORIGINAL_DATA_SUBDIRS

    train_img_dir = TRAIN_IMG_DIR
    train_anno_dir = TRAIN_ANNO_DIR

    val_img_dir = VAL_IMG_DIR
    val_anno_dir = VAL_ANNO_DIR

    test_img_dir = TEST_IMG_DIR
    test_anno_dir = TEST_ANNO_DIR

    target_img_size = TARGET_SIZE

    train_val_test_paths = divide_train_val(img_dir, anno_dir, subdirs, train_size=0.7)
    img_train_paths, img_val_paths, img_test_paths, anno_train_paths, anno_val_paths, anno_test_paths = train_val_test_paths

    save_yolo_train_val_files(img_train_paths, img_val_paths, img_test_paths,
                              anno_train_paths, anno_val_paths, anno_test_paths,
                              train_img_dir, val_img_dir, test_img_dir,
                              train_anno_dir, val_anno_dir,test_anno_dir,
                              target_img_size)

    save_yaml_yolo()


if __name__ == '__main__':
    main()

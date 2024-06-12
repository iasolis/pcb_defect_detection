from preprocess import load_bbox, load_img

import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_img(img: np.array, title: str = None) -> None:
    """
    Функция показа на экран изображения печатной платы
    Parameters
    ----------
    img: np.array
        Изображение в виде массива
    title: str
        Заголовок изображения
    """
    plt.figure(figsize=(18, 9))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.imshow(np.asarray(img), cmap='gray')
    plt.show()


def show2imgs(img1, img2, title1 = None, title2=None):
    """

    Parameters
    ----------
    img1
    img2
    title1
    title2

    Returns
    -------

    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    if title1:
        axes[0].set_title(title1)
    axes[0].axis('off')
    axes[0].imshow(img1)

    if title2:
        axes[1].set_title(title2)

    axes[1].axis('off')
    axes[1].imshow(img2)
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()

def plot_bboxes(img: np.array, bboxes: list, category_ids: list) -> None:
    """
    Функция добавления ограничивающих прямоугольников на график
    Parameters
    ----------
    img: np.array
        Изображение в виде массива
    bboxes:
        Список с параметрами ограничивающего прямоугольника
    category_ids:
        Список с номерами идентификаторов дефектов
    category_id_to_name:
        Словарь сопоставления id и имени дефекта
    """
    category_id_to_name = {0: 'spurious_copper',
                           1: 'mouse_bite',
                           2: 'open_circuit',
                           3: 'missing_hole',
                           4: 'spur',
                           5: 'short'}

    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x - w // 2), int(y - h // 2), int(x + w // 2), int(y + h // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.text(x2, y1, class_name, c='w')
    return img


def show_pcb_with_bbox(img: np.array, bboxes: list, category_ids: list, title: str) -> None:
    """
    Функция отображения изображения печатной платы вместе с ограничивающими прямоугольниками
    Parameters
    ----------
    img: np.array
        Изображение в виде массива
    bboxes:
        Список с параметрами ограничивающего прямоугольника
    category_ids:
        Список с номерами идентификаторов дефектов
    category_id_to_name:
        Словарь сопоставления id и имени дефекта
    title: str
        Заголовок изображения
    """
    plt.figure(figsize=(18, 9))
    img_w_bboxes = plot_bboxes(img, bboxes, category_ids)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_w_bboxes, cmap='gray')
    plt.show()


def show_prediction_result(img: np.array, orig_bboxes, predict_bboxes, orig_ids, predict_ids):
    """
    Функция отображающая результаты предиктов, первое изображение с реальными
    ограничивающими прямогольниками, а второе с предсказанными моделью
    Parameters
    ----------
    image
    orig_bboxes
    pred_bboxes
    orig_ids
    pred_ids
    category_id_to_name

    Returns
    -------

    """
    orig_img = img.copy()
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    orig_img = plot_bboxes(orig_img, orig_bboxes, orig_ids)
    axes[0].axis('off')
    axes[0].set_title('Original bboxes image')
    axes[0].imshow(orig_img, cmap='gray')

    predict_img = img.copy()
    predict_img = plot_bboxes(predict_img, predict_bboxes, predict_ids)
    axes[1].axis('off')
    axes[1].set_title('Predicted bboxes image')
    axes[1].imshow(predict_img, cmap='gray')
    plt.show()

def main():

    category_ids = [5, 5, 5]
    image_path = 'yolov8/datasets/data/images/test/01_spurious_copper_05.jpg'
    anno_path = 'yolov8/datasets/data/labels/test/01_spurious_copper_05.txt'
    image = load_img(image_path)
    bbox = load_bbox(anno_path) * 640
    show_pcb_with_bbox(image, bbox, category_ids, 'original_image')


if __name__ == '__main__':
    main()

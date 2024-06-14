from preprocess import load_bbox, load_img

import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_img(img: np.array, title: str = None) -> None:
    """
    Функция отображения одного изображения на экран
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


def show2imgs(img1: np.ndarray, img2: np.ndarray, title1: str = None, title2: str = None):
    """
    Функция отображения двух изображений на экран

    Parameters
    ----------
    img1: np.ndarray
        Первое отображаемое изображение
    img2: np.ndarray
        Второе отображаемое изображение
    title1: str
        Заголовок для первого изображения
    title2: str
        Заголовок для второго изображения

    Returns
    -------
    None
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


def plot_bboxes(img: np.ndarray, bboxes: list, category_ids: list) -> np.ndarray:
    """
    Функция добавления ограничивающих прямоугольников на график

    Parameters
    ----------
    img: np.array
        Изображение в виде массива
    bboxes:
        Список с параметрами ограничивающего прямоугольника
    category_ids:
        Список с номерами идентификаторов класса дефектов

     Returns
    -------
    np.ndarray
        Изображение с окружающими прямоугольниками в виде
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
        Список с параметрами ограничивающих прямоугольников
    category_ids:
        Список с номерами идентификаторов класса дефектов
    category_id_to_name:
        Словарь сопоставления id и имени дефекта
    title: str
        Заголовок изображения
    """
    plt.figure(figsize=(20, 14))
    img_w_bboxes = plot_bboxes(img, bboxes, category_ids)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_w_bboxes, cmap='gray')
    plt.show()


def show_prediction_result(img: np.ndarray, orig_bboxes: list, predict_bboxes: list,
                           orig_ids: list, predict_ids: list) -> None:
    """
    Функция отображающая результаты предиктов, первое изображение с реальными
    ограничивающими прямогольниками, а второе с предсказанными моделью

    Parameters
    ----------
    img: np.ndarray
        Изображение в виде массива
    orig_bboxes:
        Список с параметрами оригинальных ограничивающих прямоугольников
    predict_bboxes:
        Список с параметрами предсказанных ограничивающих прямоугольников
    orig_ids:
        Идентификаторы класса дефекта
    predict_ids:
        Идентификаторы класса дефекта

    Returns
    -------
    None
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
    image_path = 'yolov8/datasets/data/images/test/01_open_circuit_19.jpg'
    anno_path = 'yolov8/datasets/data/labels/test/01_open_circuit_19.txt'
    image = load_img(image_path)
    bbox = load_bbox(anno_path) * 640
    show_pcb_with_bbox(image, bbox, category_ids, 'original_image')


if __name__ == '__main__':
    main()

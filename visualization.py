import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_pcb(img: np.array, title: str) -> None:
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
    plt.title(title)
    plt.axis('off')
    plt.imshow(np.asarray(img), cmap='gray')
    plt.show()


def add_bboxes_to_graphic(img: np.array, bboxes: list, category_ids: list, category_id_to_name: dict) -> None:
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
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x - w // 2), int(y - h // 2), int(x + w // 2), int(y + h // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.text(x2, y1, class_name, c='w')


def show_pcb_with_bbox(img: np.array, bboxes: list, category_ids: list, category_id_to_name: dict, title: str) -> None:
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
    add_bboxes_to_graphic(img, bboxes, category_ids, category_id_to_name)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_prediction_result(img: np.array, orig_bboxes: list, pred_bboxes: list, orig_ids: list, pred_ids: list,
                           category_id_to_name: dict):
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
    original = 'Original image'
    predicted = 'Predicted image'
    show_pcb_with_bbox(img, orig_bboxes, orig_ids, category_id_to_name, original)
    show_pcb_with_bbox(img, pred_bboxes, pred_ids, category_id_to_name, predicted)





def load_bbox(anno_path):
    bbox = []
    with open(anno_path, 'r') as file:
        for lines in file:
            lines = list(map(float, lines.split()))[1:]
            bbox.append(lines)
    return np.array(bbox)


def main():
    category_id_to_name = {0: 'spurious_copper',
                           1: 'mouse_bite',
                           2: 'open_circuit',
                           3: 'missing_hole',
                           4: 'spur',
                           5: 'short'}
    category_ids = [5, 5, 5]
    image_path = 'yolov8/datasets/data/images/train/01_missing_hole_01.jpg'
    anno_path = 'yolov8/datasets/data/labels/train/01_missing_hole_01.txt'
    image = load_img(image_path)
    bbox = load_bbox(anno_path) * 640
    show_pcb_with_bbox(image, bbox, category_ids, category_id_to_name)


if __name__ == '__main__':
    main()

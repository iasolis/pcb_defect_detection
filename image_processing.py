import cv2
import numpy as np
from PIL import Image, ImageEnhance


def check_images_in_different_color_spaces(image: np.ndarray) -> dict:
    """
    Функция проверки отображения исходного  изображения в различных цветовые пространствах и компановка их в словарь
    Parameters
    ----------
    image : np.ndarray
        Изображение в виде матрицы чисел
    Returns
    -------
    dict
        Словарь изображений в разных цветовых пространствах, ключ словаря - цветовое пространство, значение -
        изображение в виде матрицы чисел
    """
    color_spaces = ('RGB', 'HSV', 'GRAY', 'LAB', 'XYZ', 'YUV')
    color_images = {color: cv2.cvtColor(image, getattr(cv2, 'COLOR_BGR2' + color))
                    for color in color_spaces}
    return color_images


def sobel_operator(img_gray: np.array, alpha: float, beta: float) -> np.array:
    """
    Функция обработки изображения оператором Собеля
    Parameters
    ----------
    img_gray: np.array
        Изображение в сером цветовом пространстве
    alpha:
        Вес x элементов массива, при смешивании двух градиентов по разным координатам
    beta:
        Вес x элементов массива, при смешивании двух градиентов по разным координатам
    Returns
    -------
    np.array
        Изображение обработанное оператором Собеля
    """
    grad_sobel_x = cv2.convertScaleAbs(cv2.Sobel(img_gray, cv2.CV_16S, 1, 0))
    grad_sobel_y = cv2.convertScaleAbs(cv2.Sobel(img_gray, cv2.CV_16S, 0, 1))
    grad_sobel_x = cv2.convertScaleAbs(grad_sobel_x)
    grad_sobel_y = cv2.convertScaleAbs(grad_sobel_y)
    image_sobel = cv2.addWeighted(grad_sobel_x, alpha, grad_sobel_y, beta, 0)
    return image_sobel


def canny_operator(img_gray: np.array, threshold1: float, threshold2: float) -> np.array:
    """
    Функция обработки изображения оператором Кэнни
    Parameters
    ----------
    img_gray: np.array
        Изображение в сером цветовом пространстве
    threshold1: float
        Первый порог для процедуры гистерезиса.
    threshold2: float
        Второй порог для процедуры гистерезиса.
    Returns
    -------
    np.array
        Изображение обработанное оператором Кэнни
    """
    image_canny = cv2.Canny(img_gray, threshold1, threshold2)
    return image_canny


def symmetrize_img(img: np.array, target_size: int = 640) -> np.array:
    """
    Функция для изменения разрешения изображения
    Parameters
    ----------
    img: np.array
        Изначальное изображение в виде массива np.array
    target_size: int, int
        Конечный размер изображения
    Returns
    -------
    np.array
        Изображение в виде массива np.array
    """
    img_size = (target_size, target_size)
    resized_img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    return resized_img


def image_process(img: np.array):
    img = cv2.cvtColor(img, getattr(cv2, 'COLOR_BGR2GRAY'))
    pill_image = Image.fromarray(img)
    contrast_img = ImageEnhance.Contrast(pill_image).enhance(2)
    brightness_img = ImageEnhance.Brightness(contrast_img).enhance(2)
    img = np.asarray(brightness_img)
    img = sobel_operator(img, 0.1, 0.1)
    return img


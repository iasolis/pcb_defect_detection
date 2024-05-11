TARGET_SIZE: int = 640

# for colab
ORIGINAL_DATA_DIR: str = '/content/drive/MyDrive/VKRM/pcb_defect_detection/original_data/'
ORIGINAL_DATA_IMG_DIR: str = '/content/drive/MyDrive/VKRM/pcb_defect_detection/original_data/images/'
ORIGINAL_DATA_ANNO_DIR: str = '/content/drive/MyDrive/VKRM/pcb_defect_detection/original_data/annotations/'


# ORIGINAL_DATA_DIR: str = '../original_data/'
# ORIGINAL_DATA_IMG_DIR: str = '../original_data/images/'
# ORIGINAL_DATA_ANNO_DIR: str = '../original_data/annotations/'
ORIGINAL_DATA_SUBDIRS: list[str] = ['spurious_copper/', 'mouse_bite/', 'open_circuit/', 'missing_hole/', 'spur/', 'short/']


DATA_DIR: str = 'datasets/data/'
TRAIN_IMG_DIR: str = 'datasets/data/images/train/'
TRAIN_ANNO_DIR: str = 'datasets/data/labels/train/'
VAL_IMG_DIR: str = 'datasets/data/images/val/'
VAL_ANNO_DIR: str = 'datasets/data/labels/val/'

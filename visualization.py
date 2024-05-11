import matplotlib.pyplot as plt
from os.path import basename

def plot_pcb_with_bndboxs(labels, boxes: str, img_path):
    image_name = basename(img_path)
    label = labels
    orig_img_path = f'original_data/images/{label}/image_name'

    image = image = plt.imread(orig_img_path)


    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

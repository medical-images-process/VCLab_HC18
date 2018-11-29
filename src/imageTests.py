from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np


def imageResize(image, output_size):
    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size, output_size

    #new_h, new_w = output_size, output_size
    new_h, new_w = int(new_h), int(new_w)

    new_image = transform.resize(image, (new_h, new_w))

    return new_image

def randomCorp(image, output_size):


    h, w = image.shape[:2]
    new_h, new_w = output_size -1, output_size -1

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    new_image = image[top: top + new_h,
            left: left + new_w]

    return new_image


path = 'C:/Users/danielc/Documents/Studium/Master/Visual_Computing/VCLab_HC18/Dataset/training/set/'
output_size = 512
assert isinstance(output_size, (int, tuple))
image = io.imread(path + '000_HC.png')
#img_y = io.imread(path + '000_HC_Annotation.png')

rs_img = imageResize(image, output_size=output_size)
rc_img= randomCorp(rs_img, output_size=output_size)

plt.imshow(rs_img)
plt.show()

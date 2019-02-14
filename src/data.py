from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

import os
import csv
import numpy as np
import skimage.io as io
import skimage.transform as trans

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, seg, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        seg = seg[:, :, :, 0] if (len(seg.shape) == 4) else seg[:, :, 0]
        new_seg = np.zeros(seg.shape + (num_class,))
        for i in range(num_class):
            new_seg[seg == i, i] = 1
        new_seg = np.reshape(new_seg, (new_seg.shape[0], new_seg.shape[1] * new_seg.shape[2],
                                       new_seg.shape[3])) if flag_multi_class else np.reshape(new_seg, (
            new_seg.shape[0] * new_seg.shape[1], new_seg.shape[2]))
        seg = new_seg
    elif (np.max(img) > 1):
        img = img / 255
        seg = seg / 255
        # seg[seg > 0.5] = 1
        # seg[seg <= 0.5] = 0
    return (img, seg)


def column(matrix, i):
    return [row[i] for row in matrix]


def trainGenerator(batch_size, train_path, image_folder, seg_folder, traindf, aug_dict, image_color_mode="grayscale",
                   seg_color_mode="grayscale", image_save_prefix="image", seg_save_prefix="seg",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 384), seed=1):
    '''
    can generate image and seg at the same time
    use the same seed for image_datagen and seg_datagen to ensure the transformation for image and seg is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    csv_datagen = ImageDataGenerator(**aug_dict)
    seg_datagen = ImageDataGenerator(**aug_dict)

    seg_generator = seg_datagen.flow_from_directory(
        train_path,
        classes=[seg_folder],
        class_mode=None,
        color_mode=seg_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=seg_save_prefix,
        seed=seed)
    csv_generator = csv_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=os.path.join(train_path, 'image'),
        x_col='filename',
        y_col=['cx', 'cy', 'sa', 'sb', 'sin', 'cos', 'hc'],
        class_mode="other",
        subset="training",
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    train_generator = zip(seg_generator, csv_generator)
    for (seg, img_csv) in train_generator:
        img, seg = adjustData(img_csv[0], seg, flag_multi_class, num_class)
        cx = np.array(column(img_csv[1], 0), dtype=np.float32)
        cy = np.array(column(img_csv[1], 1), dtype=np.float32)
        sa = np.array(column(img_csv[1], 2), dtype=np.float32)
        sb = np.array(column(img_csv[1], 3), dtype=np.float32)
        sin = np.array(column(img_csv[1], 4), dtype=np.float32)
        cos = np.array(column(img_csv[1], 5), dtype=np.float32)
        hc = np.array(column(img_csv[1], 6), dtype=np.float32)
        yield (img, [seg, cx, cy, sa, sb, sin, cos, hc])


def testGenerator(test_path, num_image=30, target_size=(256, 384), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%03d_HC.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    return img / 255


def saveResult(save_path, norm, npyfile, flag_multi_class=False, num_class=2):
    with open(os.path.join(save_path, 'results.csv'), "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['filename'] + ['center_x_mm'] + ['center_y_mm'] + ['semi_axes_a_mm'] + ['semi_axes_b_mm']
                        + ['angle_rad'] + ['hc_mm'])
        i = 0
        testgen = zip(npyfile[0],  npyfile[1], npyfile[2], npyfile[3], npyfile[4], npyfile[5], npyfile[6], npyfile[7])
        for (img, cx, cy, sa, sb, sin, cos, hc) in testgen:
            img = labelVisualize(num_class, COLOR_DICT, img) if flag_multi_class else img[:, :, 0]
            io.imsave(os.path.join(save_path, 'out', "%d_predict.png" % i), img)
            writer.writerow([str(i)] +
                            [str(round(cx[0] * norm['cx'] + norm['cx'], 9))] +
                            [str(round(cy[0] * norm['cy'] + norm['cy'], 9))] +
                            [str(round(sa[0] * norm['sa'] + norm['sa'], 9))] +
                            [str(round(sb[0] * norm['sb'] + norm['sb'], 9))] +
                            [str(round(np.arctan2(sin[0], cos[0]), 9))] +
                            [str(round(hc[0] * norm['hc'] + norm['hc'], 2))])
            i += 1



def preTrainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,384),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        xx = np.zeros(batch_size)
        yield (img, [mask, xx, xx, xx, xx, xx, xx, xx])


def preTestGenerator(test_path,num_image = 30,target_size = (256,384),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def preSaveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
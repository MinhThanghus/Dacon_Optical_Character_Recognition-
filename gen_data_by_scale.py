import os.path
import albumentations as A
import numpy as np
from PIL import Image
import cv2
transform = A.Compose([
            A.AdvancedBlur(rotate_limit=15, p=1)
        ])
transform2 = A.Compose([
            A.ShiftScaleRotate(rotate_limit=15, p=1)
        ])
M = np.float32([[1,0,100],[0,1,50]])

# perform the translation

folder_save = r"/home/thangnm/encode/data_gen_by_albumentation_advanced_blud"
list_image = []
with open("train.txt",encoding="utf-8") as f:
    # line = f.readlines()
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        content = line.split('\t')[1]


        list_image.append(line)
    index = 44444
    for i,image in enumerate(list_image):
        slect_image_1 = list_image[i]
        if (i+1)<len(list_image):
            lisst_slect_image_2 = list_image[i+1]
            print(slect_image_1,lisst_slect_image_2)

            img1_read = cv2.imread(slect_image_1.split('\t')[0], 1)
            img2_read = cv2.imread(lisst_slect_image_2.split('\t')[0], 1)
            height, width, _ = img2_read.shape
            M = np.float32([[1, 0, 5], [0, 1, 5]])

            # perform the translation
            img = cv2.warpAffine(img2_read, M, (width, height))

            # img1 = cv2.resize(img1_read, (100, 125))
            # img2 = cv2.resize(img2_read, (100, 125))
            file_name = str(index) + "_"+slect_image_1.split('\t')[1]+lisst_slect_image_2.split('\t')[1]+str("_.png")
            # addh = cv2.hconcat([img1, img2])
            addh = cv2.hconcat([img1_read, img])
            cv2.imwrite(os.path.join(folder_save,file_name), addh)
        index+=1
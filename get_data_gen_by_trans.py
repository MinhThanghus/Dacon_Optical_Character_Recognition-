import os.path
import albumentations as A
from PIL import Image
import cv2
transform = A.Compose([
           
          
         
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=2, p=1),
          

            A.ToGray(p=1),
        ])
transform2 = A.Compose([
            A.Blur(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=2,  p=1),
            A.ToGray(p=1),
            

           
            
        ])

folder_save = r"/home/thangnm/encode/data_gen_hsv_image_2"
list_image = []
with open("val.txt",encoding="utf-8") as f:
    # line = f.readlines()
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        content = line.split('\t')[1]
        print(len(content))
        
        list_image.append(line)
   
    index = 2111111
    for i,image in enumerate(list_image):
        slect_image_1 = list_image[i]
        if (i+1)<len(list_image):
            lisst_slect_image_2 = list_image[i+1]
            print(slect_image_1,lisst_slect_image_2)

            img1_read = cv2.imread(slect_image_1.split('\t')[0], 1)
            img2_read = cv2.imread(lisst_slect_image_2.split('\t')[0], 1)

            img1 = transform(image=img1_read)
            transformed_image_1 = img1['image']

            img2 = transform2(image=img2_read)
            transformed_image_2 = img2['image']
            

            # img1 = cv2.resize(img1_read, (100, 125))
            # img2 = cv2.resize(img2_read, (100, 125))
            file_name = str(index) + "_"+slect_image_1.split('\t')[1]+lisst_slect_image_2.split('\t')[1]+str("_.png")
            # addh = cv2.hconcat([img1, img2])
            addh = cv2.hconcat([transformed_image_1, transformed_image_2])
            cv2.imwrite(os.path.join(folder_save,file_name), addh)
        index+=1
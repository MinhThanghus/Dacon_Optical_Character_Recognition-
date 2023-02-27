import cv2
import os
folder_save = r"/home/thangnm/encode/data_gen_not_resize"
list_image = []
with open("train.txt",encoding="utf-8") as f:
    # line = f.readlines()
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        # content = line.split('\t')[1]
        # if len(line)<=3:
        list_image.append(line)
    index = 211111
    for i,image in enumerate(list_image):
        slect_image_1 = list_image[i]

        lisst_slect_image_2 = list_image[:len(list_image)]
        for i in lisst_slect_image_2:
            print(slect_image_1,i)
            img1_read = cv2.imread(slect_image_1.split('\t')[0], 1)
            img2_read = cv2.imread(i.split('\t')[0], 1)

            
            file_name = str(index) + "_"+slect_image_1.split('\t')[1]+i.split('\t')[1]+str("_.png")
            addh = cv2.hconcat([img1_read, img1_read])
            cv2.imwrite(os.path.join(folder_save,file_name), addh)
        index+=1
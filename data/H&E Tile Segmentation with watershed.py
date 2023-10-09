import cv2
import matplotlib.pyplot as plt
import math
import os
import openslide
import numpy as np
import pandas as pd
from Visual import *
from PIL import Image, ImageOps
def Image_Show(image):
    '''
    ##CV2读取文件的颜色通道为BGR,因此需要提取通道为RGB可以显示原图
    :param image: CV2格式的文件
    :return: RGB图像显示
    '''
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    plt.show()
def do_image_cut(image,resize,image_size,Cut=1,magnification=4):

    image = cv2.resize(image, dsize=(int(np.sqrt((resize*resize)//magnification)), int(np.sqrt((resize*resize)//magnification))), interpolation=cv2.INTER_LINEAR)
    ##筛选图像的阈值
    Thres_,_,_=do_thresholding(image)
    Image_Thres = (Thres_ - 15) * image_size * image_size * 3
    image_shape =image.shape
    w_num = math.ceil(image_shape[0]/image_size)
    h_num = math.ceil(image_shape[1] / image_size)
    Block = np.zeros((image_size, image_size, image.shape[2]), dtype=np.uint8)
    image_list = []
    if Cut==0:
        for i in range(w_num-1):
            for j in range(h_num-1):
                New_Image=image[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size]
                image_list.append(New_Image)
            image_list.append(image[i*image_size:(i+1)*image_size,-image_size:])
        for k in range(h_num-1):
            image_list.append(image[-image_size:, k*image_size:(k+1) * image_size])
        image_list.append(image[-image_size:, -image_size:])
    else:
        for i in range(w_num-1):
            for j in range(h_num-1):
                New_Image=image[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size]
                ##对背景色过多的图像，使用空白组代替
                image_iN = 255 - New_Image
                thres_images = np.sum(image_iN)
                if thres_images > Image_Thres:
                    New_Image=New_Image
                else:
                    New_Image=Block
                image_list.append(New_Image)
            ##对背景色过多的图像，使用空白组代替
            New_Image2=image[i * image_size:(i + 1) * image_size, -image_size:]
            image_iN = 255 - New_Image2
            thres_images = np.sum(image_iN)
            if thres_images > Image_Thres:
                New_Image2 = New_Image2
            else:
                New_Image2 = Block
            image_list.append(New_Image2)
        for k in range(h_num-1):
            New_Image3=image[-image_size:, k*image_size:(k+1) * image_size]
            image_iN = 255 - New_Image3
            thres_images = np.sum(image_iN)
            if thres_images > Image_Thres:
                New_Image3 = New_Image3
            else:
                New_Image3 = Block
            image_list.append(New_Image3)
        New_Image4 = image[-image_size:, -image_size:]
        image_iN = 255 - New_Image4
        thres_images = np.sum(image_iN)
        if thres_images > Image_Thres:
            New_Image4 = New_Image4
        else:
            New_Image4 = Block
        image_list.append(New_Image4)
    return image_list,Thres_

def mask_merge(image_list,image_shape,image_size):
    '''
    ##将切割好的图像重新拼接
    :param image_list: 存储了所有的切割图像
    :param image_shape: 将切割的图像返还的图像大小
    :param image_size: 单个切割图象大小
    :return: 拼接好的图像
    '''
    w_num = math.ceil(int(image_shape)/int(image_size))
    h_num = math.ceil(int(image_shape)/int(image_size))
    image = np.zeros((image_shape,image_shape,3), dtype=float, order='C')
    index = 0
    for i in range(w_num-1):
        for j in range(h_num-1):
            image[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size]=image_list[index]
            index+=1
        image[i * image_size:(i + 1) * image_size, (j+1)*image_size:] = image_list[index][:,-(image_shape-(j+1)*image_size):]
        index+=1
    for k in range(h_num-1):
        image[-image_size:, k*image_size:(k+1) * image_size]=image_list[index]
        index+=1
    image[(i+1) * image_size:, (k+1) * image_size:] = image_list[index][-(image_shape-(i+1) * image_size):,-(image_shape-(k+1)*image_size):]
    return image
def Add_Border(image_list,Bold_Image_Szie):
    New_List_Image=[]
    for i in image_list:
        img = cv2.copyMakeBorder(i, Bold_Image_Szie, Bold_Image_Szie, Bold_Image_Szie, Bold_Image_Szie,
                                         cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        New_List_Image.append(img)
    return New_List_Image
def do_thresholding(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # import Visual
    # thres=102
    # thres_img=np.zeros((img_c.shape[0],img_c.shape[1]),dtype=np.uint8)
    # thres_img[img_c<thres]=0
    # thres_img[img_c>=thres]=255
    # Visual.histogram(img,thres_img,img_c,thres)
    # ##阈值
    # plt.imshow(thres_img)
    # plt.show()

    return thres, thres_img, img_c
#舍弃包含图像少的样本
def image_write(save_path,image_list,thres,Orial_Name):
    new_dataframe = pd.DataFrame()
    ##Patch的阈值
    # Image_Thres=(thres-15)*image_list[0].shape[0]*image_list[0].shape[1]*3
    index_=0
    for index,image in enumerate(image_list):
        ##筛选掉背景色过多的Patch
        if np.sum(image)!=0:
            new_image_name=Orial_Name.split('.')[0]+'_'+str(index)+'.'+Orial_Name.split('.')[1]
            image_path = save_path+new_image_name
            cv2.imwrite(image_path,image)
            index_+=1
        # ##筛选掉背景色过多的Patch
        # image_iN=255-image
        # thres_images=np.sum(image_iN)
        # if thres_images>Image_Thres:
        #     new_image_name=Orial_Name.split('.')[0]+'_'+str(index)+'.'+Orial_Name.split('.')[1]
        #     image_path = save_path+new_image_name
        #     cv2.imwrite(image_path,image)
    return new_dataframe

def Tile_Image(image_list,image_shape,image_size):
    '''
    ##将切割好的图像重新拼接
    :param image_list: 存储了所有的切割图像
    :param image_shape: 将切割的图像返还的图像大小
    :param image_size: 单个切割图象大小
    :return: 拼接好的图像
    '''
    # w_num = math.ceil(int(image_shape[0])/int(image_size))
    # h_num = math.ceil(int(image_shape[1])/int(image_size))
    #
    w_num = int(image_shape[0])
    h_num =int(image_shape[1])
    image_shapes0=w_num*int(image_size)
    image_shapes1=h_num*int(image_size)
    image = np.zeros((w_num*int(image_size),h_num*int(image_size),3), dtype=float, order='C')
    index = 0
    for i in range(w_num-1):
        for j in range(h_num-1):
            image[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size]=image_list[index]
            index+=1
        image[i * image_size:(i + 1) * image_size, (j+1)*image_size:] = image_list[index][:,-(image_shapes1-(j+1)*image_size):]
        index+=1
    for k in range(h_num-1):
        image[-image_size:, k*image_size:(k+1) * image_size]=image_list[index]
        index+=1
    image[(i+1) * image_size:, (k+1) * image_size:] = image_list[index][-(image_shapes0-(i+1) * image_size):,-(image_shapes1-(k+1)*image_size):]
    return image


def main(image_path,save_path,Cut_Image_Size,Orial_Image_Size,Bold_Image_Szie=1,magnification=4):
    image = cv2.imread(image_path)
    image=cv2.resize(image,dsize=(Orial_Image_Size,Orial_Image_Size),interpolation=cv2.INTER_LINEAR)
    ##对整个图像进行切割
    image_list,thres=do_image_cut(image,resize=Orial_Image_Size,image_size=Cut_Image_Size,magnification=magnification)
    ##保存切割的图像（并以thres筛选背景色的图像）
    # image_write(save_path,image_list,thres,Orial_Name=image_path.split('/')[-1])
    ##对图像添加Border
    Border_List=Add_Border(image_list,Bold_Image_Szie)
    ##平铺Tiles
    Tile_list=[]
    for bord in Border_List:
        if np.sum(bord)!=0:
            bord[bord == 0] = 255
            Tile_list.append(bord)
    image_shape=[7,12]
    image_size = Cut_Image_Size + Bold_Image_Szie * 2
    Image_Tile=Tile_Image(Tile_list, image_shape, image_size)
    Image_Show(Image_Tile)

    # ##对切割的图像进行重新拼接
    Imagess=mask_merge(Border_List,
                       int((Cut_Image_Size+Bold_Image_Szie*2)*(int(np.sqrt((Orial_Image_Size*Orial_Image_Size)//magnification))/Cut_Image_Size)),
                       Cut_Image_Size+Bold_Image_Szie*2)
    # ##显示单张图像
    Image_Show(Imagess)
    a=1


def WSI_Whole(image_path,outPut):
    image = cv2.imread(image_path)
    image=cv2.resize(image,dsize=(384,384),interpolation=cv2.INTER_LINEAR)
    # b, g, r = cv2.split(image)
    # image = cv2.merge([r, g, b])

    # cv2.imwrite(outPut+image_path.split('/')[-1], image)


    # plt.imshow(image)
    # plt.show()


if __name__ == '__main__':

    # a=1
    #
    #
    # Image=10000*10000
    # len=np.sqrt(Image//4)
    #
    #


    save_path= '../../../Datasets/Image_4_magnification/'
    Paths='../../../Datasets/SCLC_Image/'
    # main(image_path='../../../Datasets/Image_4_magnification/Zhou-1_0_34.png', save_path=save_path, Cut_Image_Size=256, Orial_Image_Size=10000, Bold_Image_Szie=10,
    #      magnification=4)

    FileName=os.listdir(Paths)
    for name in FileName:
        print('已经输出:'+name)
        Image_Patch=Paths+name+'/'
        Image_Name=os.listdir(Image_Patch)
        for images in Image_Name:
            image_path =Image_Patch+images
            # WSI_Whole(image_path,save_path)
            main(image_path=image_path,save_path=save_path,Cut_Image_Size=224,Orial_Image_Size=10000,Bold_Image_Szie=10,magnification=4)

import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
def Image_Show(image):


    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    plt.axis('off')
    plt.savefig(Filename)
    plt.show()
def Add_Border(image_list,Bold_Image_Szie):
    New_List_Image=[]
    for i in image_list:
        img = cv2.copyMakeBorder(i, Bold_Image_Szie, Bold_Image_Szie, Bold_Image_Szie, Bold_Image_Szie,
                                         cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))
        New_List_Image.append(img)
    return New_List_Image

def mask_merge(image_list,image_shape,image_size):
    '''
    ##将切割好的图像重新拼接
    :param image_list: 存储了所有的切割图像
    :param image_shape: 将切割的图像返还的图像大小
    :param image_size: 单个切割图象大小
    :return: 拼接好的图像
    '''

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

def Picture(Cut_Image_Size,Bold_Linner_Szie,Row_COL,Low_patient_Path,Height_patient_Path,Choice=1):
    '''
    :param Cut_Image_Size: 将图像缩放到该尺寸
    :param Bold_Linner_Szie:每个图像的边框大小
    :param Row_COL: 输出nxn的图像
    :param Low_patient_Path: 低风险Patch块的路径
    :param Height_patient_Path: 高风险Patch块的路径
    :param Choice: 1为低风险 2为高风险
    :return: ImageShow
    '''
    Low_image_List=[]
    Height_image_List=[]
    for i in range(0,Row_COL**2):
        Low_image = cv2.imread(Low_patient_Path[i])
        Low_image=cv2.resize(Low_image,(Cut_Image_Size,Cut_Image_Size),interpolation=cv2.INTER_LINEAR)
        Low_image_List.append(Low_image)
        # image_path_Low = 'Low_Risk_Patch/' + Low_patient[i] + '.png'
        # cv2.imwrite(image_path_Low,Low_image)
        Height_image = cv2.imread(Height_patient_Path[i])
        Height_image = cv2.resize(Height_image, (Cut_Image_Size, Cut_Image_Size), interpolation=cv2.INTER_LINEAR)
        Height_image_List.append(Height_image)
        # image_path_Height = 'Height_Risk_Patch/' + Height_patient[i] + '.png'
        # cv2.imwrite(image_path_Height,Height_image)
    if Choice==1:
        Border_List = Add_Border(Low_image_List, Bold_Image_Szie=Bold_Linner_Szie)
    else:
        Border_List = Add_Border(Height_image_List, Bold_Image_Szie=Bold_Linner_Szie)
    Border_Size = Cut_Image_Size + Bold_Linner_Szie * 2
    Image_Concat_Size = int(Border_Size * Row_COL)
    Imagess = mask_merge(Border_List, Image_Concat_Size, Border_Size)
    return Imagess
def Datasets_create():
    Train_Check=pd.read_csv("Path",sep=',')
    Train=pd.read_csv("Path",sep=',').iloc[:,[1,-1]]
    Test_Check=pd.read_csv("Path",sep=',')
    Test=pd.read_csv("Path",sep=',').iloc[:,[1,-1]]
    Comp_Check=pd.read_csv("Path",sep=',')
    Comp=pd.read_csv("Path",sep=',').iloc[:,[1,-1]]
    ZC_Check=pd.read_csv("Path",sep=',')
    ZC=pd.read_csv("Path",sep=',').iloc[:,[1,-1]]

    Data_Visual = pd.concat([Train, Test,Comp,ZC])
    Data_Check = pd.concat([Train_Check, Test_Check,Comp_Check,ZC_Check])
    # return Data_Visual,Data_Check
    return Train, Train_Check
def do_create_color():
    uni_cox=pd.read_csv("Path",sep=',')

    cluster=[]
    cluster1=[cluster.append(clu.split('.')[-1]) for clu in uni_cox.values[:,1]]
    cluster=np.array(cluster,dtype=int)
    ids=set(np.arange(0,50)).difference(set(cluster))
    HR=uni_cox.values[:,2]

    High_HR=np.array(cluster[HR>1])

    Low_HR=np.array(cluster[HR<1])

    Media_HR = np.array(cluster[HR==1])

    HML_HR=np.hstack((High_HR,Media_HR,Low_HR))

    return HML_HR

if __name__ == '__main__':

    HR_High_Low=do_create_color()
    HR_High_Low=[19,21,20,39]
    # HR_High_Low = [19]
    for HR_index in HR_High_Low:

        Train,Train_Check=Datasets_create()
        Pre_Dict=Train.values[:,-1]
        patient=Train.iloc[HR_index==Pre_Dict]['Name'].values
        Image_Path='Patch_Path/'

        patient_Path=Image_Path+patient+'.png'
        # Height_patient_Path=Image_Path+patient+'.png'
        Cut_Image_Size=256
        Row_ =16
        COL_=4
        image_shape=[COL_,Row_]
        Bold_Linner_Szie=3

        path = 'HR_Image/'+str(HR_index)
        if os.path.exists(path)==False:
            os.mkdir(path)

        image_List = []
        try:
            for i in range(0, Row_*COL_):
                indexs = np.arange(0, len(patient_Path))
                np.random.shuffle(indexs)
                image = cv2.imread(patient_Path[indexs[i]])
                image = cv2.resize(image, (Cut_Image_Size, Cut_Image_Size), interpolation=cv2.INTER_LINEAR)
                image_List.append(image)
            Border_List = Add_Border(image_List, Bold_Image_Szie=Bold_Linner_Szie)
            Border_Size = Cut_Image_Size + Bold_Linner_Szie * 2
            # Image_Concat_Size = int(Border_Size * Row_COL)
            Imagess = mask_merge(Border_List, image_shape, Border_Size)
            b, g, r = cv2.split(Imagess)
            image = cv2.merge([r, g, b])
            plt.imshow(image / 255)
            plt.axis('off')
            plt.savefig(path +"/"+ str(HR_index) + '.png', dpi=1500)
            plt.show()
        except IndexError:
            print("Error")
            continue


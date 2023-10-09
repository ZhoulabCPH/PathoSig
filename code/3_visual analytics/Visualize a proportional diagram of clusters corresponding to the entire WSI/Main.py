import random

import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import numpy as np
from matplotlib import cm as mplcm
import warnings
import os
warnings.filterwarnings('ignore')
def get_distinct_colors(n):

    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))
    return colors
def Image_Show(image):
    '''
    ##CV2读取文件的颜色通道为BGR,因此需要提取通道为RGB可以显示原图
    :param image: CV2格式的文件
    :return: RGB图像显示
    '''
    # b, g, r = cv2.split(image)
    # image = cv2.merge([r, g, b])
    # plt.imshow(image / 255)
    plt.matshow(image)
    plt.axis('off')
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
def do_mask_image(WSI_Patch,colors):


    ##将整个图片绘制成一个图块
    ID=0
    ID_list=[]
    Image_List=[]
    for index in range(529):
        if index not in Orial_ID:
            ID=0
            Image=np.zeros((224,224,3))+1
        else:
            ID=index
            Image = np.zeros((224,224,3)) + colors[int(WSI_Patch[WSI_Patch[:, 0] == index, 1])]
            # plt.imshow(Image)
            # plt.show()
            a=1
        ID_list.append(ID)
        Image_List.append(Image)
    ID_list=np.array(ID_list).reshape(23,23)
    ##对Patch设置边框
    Border_List = Add_Border(Image_List, Bold_Image_Szie)
    ##对边框patch合并成一个图像
    Imagess=mask_merge(Border_List,
                       int((Cut_Image_Size+Bold_Image_Szie*2)*(int(np.sqrt((Orial_Image_Size*Orial_Image_Size)//magnification))/Cut_Image_Size)),
                       Cut_Image_Size+Bold_Image_Szie*2)

    return Imagess,ID_list
def do_orial_image(WSI_Patch_Height,):

    ##将整个图片绘制成一个图块
    Image_List=[]
    for index in range(529):
        if index not in Orial_ID:
            Image=np.zeros((224,224,3),dtype=np.uint8)+255
        else:
            Path=Image_Path+WSI_Patch_Height.values[0,0]+'_'+str(index)+'.png'
            Image=cv2.imread(Path,cv2.IMREAD_COLOR)
            ##调整RGB图层/255
            b, g, r = cv2.split(Image)
            Image = cv2.merge([r, g, b])
            # if index == 100:
            #     Image=np.zeros((224,224,3),dtype=np.uint8)+0
        Image_List.append(Image)
    ##对Patch设置边框
    Border_List = Add_Border(Image_List, Bold_Image_Szie)
    ##对边框patch合并成一个图像
    Imagess=mask_merge(Border_List,
                       int((Cut_Image_Size+Bold_Image_Szie*2)*(int(np.sqrt((Orial_Image_Size*Orial_Image_Size)//magnification))/Cut_Image_Size)),
                       Cut_Image_Size+Bold_Image_Szie*2)
    return Imagess

def do_create_color(Step=5):
    uni_cox=pd.read_csv("./Datasets/OS_uni_cox.csv",sep=',')

    cluster=[]
    cluster1=[cluster.append(clu.split('.')[-1]) for clu in uni_cox.values[:,1]]
    cluster=np.array(cluster,dtype=int)
    ids=set(np.arange(0,50)).difference(set(cluster))
    HR=uni_cox.values[:,2]



    # cm1 = mplcm.get_cmap('jet')coolwarm/seismic/bwr
    cm1 = mplcm.get_cmap('bwr')
    ##HR>1的簇类别有
    High_HR=np.array(cluster[HR>1])
    High_Color=[]
    for i in range(len(High_HR)):
         High_Color.append(cm1(256-i*(256-125)//len(High_HR))[0:3])
    Height_Color = np.array(High_Color)
    ##HR=1或者HR=1的簇类别有
    Media_HR = np.array(cluster[HR==1])
    Media_Color=[]
    for i in range(len(Media_HR)):
         A=np.arange(10)
         random.shuffle(A)
         if(A[0]%2==0):
             Media_Color.append(cm1(123)[0:3])
         else:
                Media_Color.append(cm1(131)[0:3])

    Media_Color = np.array(Media_Color)

    ##HR<1的簇类别有
    Low_HR=np.array(cluster[HR<1])
    Low_Color=[]
    for i in range(len(Low_HR)):
         Low_Color.append(cm1(125-i*(125)//len(Low_HR))[0:3])
    Low_Color=np.array(Low_Color)




    ##HR簇整理
    HML_HR=np.hstack((High_HR,Media_HR,Low_HR))
    HML_Color = np.vstack((Height_Color, Media_Color, Low_Color))
    data_count = dict(zip(HML_HR, HML_Color))
    return HML_HR,data_count
def Wsi_id(DATA):
    nameWSI=[]
    idWSI=[]
    for names in DATA.values[:,0]:
        Nam_list=names.split('_')
        nameWSI.append(Nam_list[0]+'_'+Nam_list[1])
        idWSI.append(Nam_list[2])
    nameWSI=np.array(nameWSI)
    idWSI=np.array(idWSI,dtype=np.int)
    Datas=pd.DataFrame({'Sample_Name':nameWSI, 'Patch_Id':idWSI, 'Patch_Cluster':DATA.values[:,-1]})

    return Datas
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
    Data_Patch_id=Wsi_id(Data_Visual)

    return Data_Visual,Data_Check,Data_Patch_id


if __name__ == '__main__':
    Data_Visual,Data_Check,Data_Patch_id=Datasets_create()
    HML_HR,data_count=do_create_color(Step=5)

    Orial_Image_Size=10000
    magnification=4
    Cut_Image_Size = 224
    Bold_Image_Szie = 2
    Image_Path = 'Path/'

    WSI_Patchs = Data_Patch_id
    Train_Patch=Data_Visual
    Train_nameWSI=[]
    for names in Train_Patch.values[:,0]:
        Nam_list=names.split('_')
        Train_nameWSI.append(Nam_list[0]+'_'+Nam_list[1])
    Train_nameWSI=np.array(Train_nameWSI)
    Train_WSI=Data_Check
    HML_HR=[49]

    for HR_id in HML_HR:
        try:
            WSI = Train_WSI.loc[HR_id == Train_WSI['Label_Pre'].values].values[0,1]
        except IndexError:
            continue
        WSI_Patch_Height = WSI_Patchs.loc[WSI == Train_nameWSI]

        path = 'HR_Image/'+str(HR_id)
        if os.path.exists(path)==False:
            os.mkdir(path)
        WSI_Patch=np.array(WSI_Patch_Height.values[:, 1:3],dtype=np.int32)
        Orial_ID = np.array(WSI_Patch_Height.values[:, 1], dtype=np.int32)


        unique, count = np.unique(WSI_Patch_Height.values[:, -1], return_counts=True)
        ID=np.argsort(-count)
        Sig_Index=[19,20,21,39]

        unique=unique[ID]
        colors=get_distinct_colors(len(unique))

        # data_count = dict(zip(unique, colors))
        mask,ID_list=do_mask_image(WSI_Patch=WSI_Patch,colors=data_count)
        Image = do_orial_image(WSI_Patch_Height=WSI_Patch_Height)
        plt.imshow(Image/255.0)
        plt.savefig(path + "/1-" + WSI + '.pdf', dpi=1000)
        plt.show()


        plt.imshow(mask)
        plt.imshow(mask, cmap='seismic', alpha=1)
        plt.savefig(path +"/"+ WSI + '.pdf', dpi=1000)
        plt.show()
        print(WSI+"已经输入完成！")


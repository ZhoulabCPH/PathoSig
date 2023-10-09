import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def Clincial_Name(Clinical_Data):
    Clincial_Name=Clinical_Data.values[:,1]
    Clincial_name_List=[]
    for name in Clincial_Name:
        Clincial_name_List.append(name.split('-')[1]+'-'+name.split('-')[2])
    return np.array(Clincial_name_List)

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
std=StandardScaler()
def Seaborn_Corr(df):
    x=df
    # x = pd.DataFrame(np.random.rand(100, 8))
    #
    # 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
    vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
    sum(vif)
    eigenvalue, featurevector = np.linalg.eig(x)
    tol = [1. / variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
    correlation_table = df.corr()
    # 绘制相关性矩阵热力图
    sns.heatmap(correlation_table)
    plt.show()


def KM_File(Path):
    My_Val_Result = pd.read_csv(Path).iloc[:, [1, -1]]
    Clinical_Data=pd.read_excel('../../Datasets/Clinical_Data/20220802-342-247-48临床病理信息表(编码) .xlsx',sheet_name=0)
    names = My_Val_Result.values[:,0]
    Paint_Names = np.array([name.split('_')[0]+"_"+name.split('_')[1] for name in names])
    Paint_Type = np.array(list(set(Paint_Names)))
    Clincial_name=Clincial_Name(Clinical_Data)

    Val_OS=[]
    Val_OSState=[]
    Val_DFS=[]
    Val_Val_DFState=[]


    Val_Name=[]
    Clusterfile=np.zeros((len(Paint_Type),50),dtype=np.float32)
    for index,name_ in enumerate(Paint_Type):
        Val_Name.append(name_)
        OS=Clinical_Data.iloc[name_.split("_")[0]==Clincial_name]['OS'].values/12.0
        OSState = Clinical_Data.iloc[name_.split("_")[0]== Clincial_name]['OSState'].values
        DFS=Clinical_Data.iloc[name_.split("_")[0]==Clincial_name]['DFS'].values/12.0
        DFSState = Clinical_Data.iloc[name_.split("_")[0] == Clincial_name]['DFSState'].values

        # name_=Paint_Type[0]
        Single_WSI = np.where(name_ == Paint_Names)[0]
        for cluster in My_Val_Result.iloc[Single_WSI, 1]:
            Clusterfile[index,cluster]+=1
        Val_OS.append(OS)
        Val_OSState.append(OSState)
        Val_DFS.append(DFS)
        Val_Val_DFState.append(DFSState)
    Label_Pre = np.array(np.argmax(Clusterfile,axis=1))
    Val_OS = np.array(Val_OS)
    Val_OSState = np.array(Val_OSState)
    Val_DFS=np.array(Val_DFS)
    Val_Val_DFState=np.array(Val_Val_DFState)

    Val_Name=np.array(Val_Name)
    # Clusterfile = Clusterfile + 1
    Clusterfiles=np.sum(Clusterfile,axis=1)
    for index,i in enumerate(Clusterfiles):
        Clusterfile[index,:]=Clusterfile[index,:]/i
    # np.sum(Clusterfiless[2,:])
    # if Path.split('_')[1].split('/')[1]=='Train':#
    #     Clusterfile=std.fit_transform(Clusterfile)
    # else:
    #     Clusterfile = std.transform(Clusterfile)
    Clinical_File=pd.DataFrame({'Sample_Name':Val_Name,'Val_OSState':Val_OSState.T[0],
                  'Val_OS':Val_OS.T[0],'Val_DFS':Val_DFS.T[0],'Val_DFState':Val_Val_DFState.T[0],
                                'Label_Pre':Label_Pre})
    Index_cluster = ['Cluster:{}'.format(i) for i in range(50)]
    Index = np.hstack((np.array(["Sample_Name"]), np.array(Index_cluster)))
    Cluster_Value=np.hstack((np.array(Val_Name).reshape(-1,1),Clusterfile))
    files=pd.DataFrame(Cluster_Value,columns=Index)
    MyFile=pd.merge(files,Clinical_File,how='inner',on='Sample_Name')
    MyFile.to_csv(Path.split('_')[0].split('/')[1]+'/KM_File/'+Path.split('_')[1].split('/')[1]+Path.split('_')[-2]+'_UmapKM_BoxPlot.csv')
def WSI_Patch(Path):
    My_Val_Result = pd.read_csv(Path).iloc[:, 1:-1]
    Clinical_Data=pd.read_excel('../../../../Datasets/Clinical_Data/20220802-342-247-48临床病理信息表(编码) .xlsx',sheet_name=0)
    names = My_Val_Result.values[:,0]
    Paint_Names = np.array([name.split('_')[0] + '_' + name.split('_')[1] for name in names])
    Paint_Type = np.array(list(set(Paint_Names)))
    Clincial_name=Clincial_Name(Clinical_Data)
    Val_OS=[]
    Val_OSState=[]
    Val_Name=[]
    Clusterfile=np.zeros((len(Paint_Type),50),dtype=np.float32)
    for index,name_ in enumerate(Paint_Type):
        Val_Name.append(name_)
        OS=Clinical_Data.iloc[name_.split('_')[0]==Clincial_name]['OS'].values
        OSState = Clinical_Data.iloc[name_.split('_')[0] == Clincial_name]['OSState'].values
        # name_=Paint_Type[0]
        Single_WSI = np.where(name_ == Paint_Names)[0]
        sig_feature=np.array(My_Val_Result.values[Single_WSI, 1:],dtype=np.float32)

        Clusterfile[index, :] = np.sum(sig_feature, axis=0)
        Clusterfile[index,:] = np.sum(sig_feature, axis=0)/len(Single_WSI)

        # np.sum(Clusterfile)

        Val_OS.append(OS)
        Val_OSState.append(OSState)
    Label_Pre = np.array(np.argmax(Clusterfile,axis=1))
    Val_OS = np.array(Val_OS)
    Val_OSState = np.array(Val_OSState)
    Val_Name=np.array(Val_Name)
    # Clusterfiles=np.sum(Clusterfile,axis=1)
    # for index,i in enumerate(Clusterfiles):
    #     Clusterfile[index,:]=Clusterfile[index,:]/i
    # # Clusterfile=np.exp(1+Clusterfile)
    # np.sum(Clusterfiless[2,:])
    # if Path.split('_')[0].split('/')[-1]=='Train':
    #     Clusterfile=std.fit_transform(Clusterfile)
    # else:
    #     Clusterfile = std.transform(Clusterfile)

    Clinical_File=pd.DataFrame({'Sample_Name':Val_Name,'Val_OSState':Val_OSState.T[0],
                  'Val_OS':Val_OS.T[0],'Label_Pre':Label_Pre})
    Index_cluster = ['Cluster:{}'.format(i) for i in range(50)]
    Index = np.hstack((np.array(["Sample_Name"]), np.array(Index_cluster)))
    Cluster_Value=np.hstack((np.array(Val_Name).reshape(-1,1),Clusterfile))
    files=pd.DataFrame(Cluster_Value,columns=Index)
    MyFile=pd.merge(files,Clinical_File,how='inner',on='Sample_Name')
    MyFile.to_csv(Path.split('_')[0]+'_UmapKM_Cluster_BoxPlot.csv')

if __name__ == '__main__':
    for i in range(9,10):
        if int(i)>=10:
            Train_path = './输入文件/Umap_File/Train_' + '000000' + str(i) + '_Umap.csv'
            Test_path = './输入文件/Umap_File/Test_' + '000000' + str(i) + '_Umap.csv'
            ZC_type_path='./输入文件/Umap_File/ZC_type_' +'000000'+str(i)+'_Umap.csv'
            Composite_path = './输入文件/Umap_File/Composite_type_' + '000000' + str(i) + '_Umap.csv'
        else:
            Train_path = './输入文件/Umap_File/Train_' + '0000000' + str(i) + '_Umap.csv'
            Test_path = './输入文件/Umap_File/Test_' + '0000000' + str(i) + '_Umap.csv'
            ZC_type_path='./输入文件/Umap_File/ZC_type_' +'0000000'+str(i)+'_Umap.csv'
            Composite_path = './输入文件/Umap_File/Composite_type_' + '0000000' + str(i) + '_Umap.csv'
        KM_File(Path=Train_path)
        KM_File(Path=Test_path)
        KM_File(Path=ZC_type_path)
        KM_File(Path=Composite_path)
        print("have done "+str(i))

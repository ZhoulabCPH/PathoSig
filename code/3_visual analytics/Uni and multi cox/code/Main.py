import pandas as pd
import numpy as np

uni_cox=pd.read_csv("../data/uni_cox.csv").drop(columns='Unnamed: 0')
mul_cox=pd.read_csv("../data/mul_cox.csv")
mul_cox.columns=uni_cox.axes[1]
Univariate=[]

for values in uni_cox.values[:,1:3]:
    Univariate.append('%.2f' % values[0]+" ("+'%.2f' % float(values[1].split(" ")[0])+'-'+'%.2f' % float(values[1].split(" ")[2])+")")
uni_cox['Univariate']=np.array(Univariate)
uni_coxs=uni_cox.iloc[:,[0,4,3]]
Multivariate=[]
for values in mul_cox.values[:,1:3]:
    Multivariate.append('%.2f' % values[0]+" ("+'%.2f' % float(values[1].split(" ")[0])+'-'+'%.2f' % float(values[1].split(" ")[2])+")")
mul_cox['Multivariate']=np.array(Multivariate)
mul_coxs=mul_cox.iloc[:,[0,4,3]]

uni_mul_cox=pd.merge(uni_coxs,mul_coxs,how='outer',on='Characteristics',)
uni_mul_cox=uni_mul_cox.fillna(' ')
Colname=['Factors','Univariate  analysis HR(95% CI)','P-value','Multivariate analysis HR(95% CI)','P-value']
uni_mul_cox.columns=Colname

Index=np.argsort(uni_cox.values[:,1])
uni_mul_cox=uni_mul_cox.iloc[Index,:]


id_name=[]
for name in uni_mul_cox.values[:,0]:
    id_name.append(name.split('.')[1])
uni_mul_cox['Factors']=id_name

uni_mul_cox.to_csv("../data/Univariate_Multivariate_COX.csv",index=None)

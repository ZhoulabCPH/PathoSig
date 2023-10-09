library(Rmisc)
library(ggpubr)
library(ggplot2)
library(dplyr)
library(survival)
library(survminer)
library(cutoff)
library(reshape2)
library(forestplot)
library(zCompositions)
library(plyr)
library(tibble)
library(ggalluvial)
library(cowplot)
library(RColorBrewer)
rm(list = ls()) 
############################Recurrence####################################
setwd("Path")
Discover<- read.csv('Discover_Traditional.csv')
External1<- read.csv('External_1_Traditional.csv')
External2<- read.csv('External_2_Traditional.csv')
Neoadjuvant<- read.csv('Neoadjuvant_T.csv')

File_name=c("Discover_Traditional.csv","External_1_Traditional.csv","External_2_Traditional.csv","Neoadjuvant_T.csv")
Plots<-function(Files){
  Recurrent_DATA<- read.csv(Files)
  
  ##Del. NaN
  Recurrent<-na.omit(Recurrent_DATA)[,c(1,5,6)]
  
  Recurrent$Sig_Response=0
  Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$Val_DFState=="1"]="High & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$Val_DFState=="0"]="High & no recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$Val_DFState=="1"]="Low & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$Val_DFState=="0"]="Low & no recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$Val_DFState=="1"]="Heterogeneity & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$Val_DFState=="0"]="Heterogeneity & no recurrent"
  
  
  High_rec=sum(Recurrent$Sig_Response=="High & recurrent")
  High_Nrec=sum(Recurrent$Sig_Response=="High & no recurrent")
  Low_rec=sum(Recurrent$Sig_Response=="Low & recurrent")
  Low_Nrec=sum(Recurrent$Sig_Response=="Low & no recurrent")
  Heterogeneity_rec=sum(Recurrent$Sig_Response=="Heterogeneity & recurrent")
  Heterogeneity_Nrec=sum(Recurrent$Sig_Response=="Heterogeneity & no recurrent")
  
  
  
  High_rec_rate=High_rec/(High_rec+High_Nrec)
  High_Nrec_rate=High_Nrec/(High_rec+High_Nrec)
  Low_rec_rate=Low_rec/(Low_rec+Low_Nrec)
  Low_Nrec_rate=Low_Nrec/(Low_rec+Low_Nrec)
  
  Heterogeneity_rec_rate=Heterogeneity_rec/(Heterogeneity_rec+Heterogeneity_Nrec)
  Heterogeneity_Nrec_rate=Heterogeneity_Nrec/(Heterogeneity_rec+Heterogeneity_Nrec)
  
  
  
  confusion_ = data.frame(
    Group = c("High Risk", "Low Risk","Heterogeneity"),
    Recurrent = c(High_rec_rate,Low_rec_rate,Heterogeneity_rec_rate),
    NoRecurrent = c(High_Nrec_rate, Low_Nrec_rate,Heterogeneity_Nrec_rate)
    
  )
  confusion_dfm = melt(confusion_, id.vars=c("Group"), measure.vars=c("Recurrent","NoRecurrent"),
                       variable.name="Response", value.name="Frequency")
  confusion_dfm$Group= factor(confusion_dfm$Group, levels = confusion_$Group)
  confusion_dfm$Frequency=round(confusion_dfm$Frequency, digits = 3)
  colors<-rev(c('#d5ead9','#f5dfc6' ))
  ggplot(confusion_dfm, aes(x = Group, y = Frequency, group = Response)) + 
    geom_col(aes(fill=Response)) +
    theme(axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.25)) +
    geom_text(aes(label = Frequency), position = position_stack(vjust = .5), size = 3)+theme_bw()+ 
    theme(panel.grid=element_blank(),
          axis.text = element_text(size = 8,colour = 'black',family="Arial"),
          axis.title = element_text(size = 12,colour = 'black',family="Arial"))+
    scale_fill_manual(values=colors)+theme_classic()

}

##Del. NaN
tt="Traditional_Discover"
P1=Plots(File_name[1])
tt="Traditional_External1"
P2=Plots(File_name[2])
tt="Traditional_External2"
P3=Plots(File_name[3])
tt="Neoadjuvant"
P4=Plots(File_name[4])
setwd("Path")
pdf("Recurrence.pdf")
plot_grid(P1, P2,P3,P4,labels = c("A", "B","C", "D"),ncol = 2,nrow = 2)
dev.off() 








Plots_Max<-function(Files){
  Recurrent_DATA<- read.csv(Files)
  Recurrent<-na.omit(Recurrent_DATA)[,c(1,5,6)]
  
  Recurrent$Sig_Response=0
  Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$Val_DFState=="1"]="High & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$Val_DFState=="0"]="High & no recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$Val_DFState=="1"]="Low & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$Val_DFState=="0"]="Low & no recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$Val_DFState=="1"]="Heterogeneity & recurrent"
  Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$Val_DFState=="0"]="Heterogeneity & no recurrent"
  
  
  High_rec=sum(Recurrent$Sig_Response=="High & recurrent")
  High_Nrec=sum(Recurrent$Sig_Response=="High & no recurrent")
  Low_rec=sum(Recurrent$Sig_Response=="Low & recurrent")
  Low_Nrec=sum(Recurrent$Sig_Response=="Low & no recurrent")
  Heterogeneity_rec=sum(Recurrent$Sig_Response=="Heterogeneity & recurrent")
  Heterogeneity_Nrec=sum(Recurrent$Sig_Response=="Heterogeneity & no recurrent")
  
  confusion_ = data.frame(
    Group = c("High Risk","Heterogeneity", "Low Risk"),
    Recurrence = c(High_rec,Heterogeneity_rec,Low_rec),
    Norecurrence = c(High_Nrec,Heterogeneity_Nrec ,Low_Nrec)
    
  )
  

  conf_matrix_df <- data.frame(Recurrence = c(High_rec, Heterogeneity_rec, Low_rec),
                               Norecurrence = c(High_Nrec, Heterogeneity_Nrec, Low_Nrec),
                               row.names = c("High Risk", "Heterogeneity", "Low Risk"))
  print(paste0("fisher P:",fisher.test(confusion_[1:3,2:3])$p.value))
  

  conf_matrix <- as.matrix(conf_matrix_df)
  
  # Convert the matrix to a data frame in long format
  conf_matrix_df_long <- melt(conf_matrix)
  ggplot(conf_matrix_df_long, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = value), color = "black", size = 12) +
    theme_bw() +
    labs(x = "Actual", y = "Predicted", title = "Confusion Matrix")
  
  }

tt="Traditional_Discover"
P1=Plots_Max(File_name[1])
tt="Traditional_External1"
P2=Plots_Max(File_name[2])
tt="Traditional_External2"
P3=Plots_Max(File_name[3])
tt="Neoadjuvant"
P4=Plots_Max(File_name[4])
setwd("Path")
pdf("Name.pdf")
plot_grid(P1, P2,P3,P4,labels = c("A", "B","C", "D"),ncol = 2,nrow = 2)
dev.off() 


setwd("Path")

Cluster<- read.csv('cluster.csv')
Recurrent<-Cluster
Recurrent$cluster_ITH
Recurrent$Sig_Response=0
Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$cluster_ITH=="M-H/L-H"]="High & ML"
Recurrent$Sig_Response[Recurrent$Pre_Label=="High_Risk" & Recurrent$cluster_ITH=="H-H/Complex"]="High & HC"


Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$cluster_ITH=="M-H/L-H"]="Low & ML"
Recurrent$Sig_Response[Recurrent$Pre_Label=="Low_Risk" & Recurrent$cluster_ITH=="H-H/Complex"]="Low & HC"


Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$cluster_ITH=="M-H/L-H"]="Heterogeneity & ML"
Recurrent$Sig_Response[Recurrent$Pre_Label=="Heterogeneity" & Recurrent$cluster_ITH=="H-H/Complex"]="Heterogeneity & HC"


High_ML=sum(Recurrent$Sig_Response=="High & ML")
High_HC=sum(Recurrent$Sig_Response=="High & HC")

Low_ML=sum(Recurrent$Sig_Response=="Low & ML")
Low_HC=sum(Recurrent$Sig_Response=="Low & HC")

Heterogeneity_ML=sum(Recurrent$Sig_Response=="Heterogeneity & ML")
Heterogeneity_HC=sum(Recurrent$Sig_Response=="Heterogeneity & HC")


ML_High_rate=High_ML/(High_ML+Low_ML+Heterogeneity_ML)
ML_Low_rate=Low_ML/(High_ML+Low_ML+Heterogeneity_ML)
ML_Heterogeneity_rate=Heterogeneity_ML/(High_ML+Low_ML+Heterogeneity_ML)



HC_High_rate=High_HC/(High_HC+Low_HC+Heterogeneity_HC)
HC_Low_rate=Low_HC/(High_HC+Low_HC+Heterogeneity_HC)
HC_Heterogeneity_rate=Heterogeneity_HC/(High_HC+Low_HC+Heterogeneity_HC)


confusion_ = data.frame(
  Group = c("ML", "HC"),
  High_risk = c(ML_High_rate,HC_High_rate),
  Low_risk = c(ML_Low_rate, HC_Low_rate),
  Heterogeneity = c(ML_Heterogeneity_rate, HC_Heterogeneity_rate)
)


confusion_dfm = melt(confusion_, id.vars=c("Group"), measure.vars=c("High_risk","Low_risk","Heterogeneity"),
                     variable.name="Response", value.name="Frequency")
confusion_dfm$Group= factor(confusion_dfm$Group, levels = confusion_$Group)
confusion_dfm$Frequency=round(confusion_dfm$Frequency, digits = 3)

ggplot(confusion_dfm, aes(x = Group, y = Frequency, group = Response)) + 
  geom_col(aes(fill=Response)) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.25)) +
  geom_text(aes(label = Frequency), position = position_stack(vjust = .5), size = 3)+theme_bw()+ 
  theme(panel.grid=element_blank(),
        axis.text = element_text(size = 8,colour = 'black',family="Arial"),
        axis.title = element_text(size = 12,colour = 'black',family="Arial"))+
  scale_fill_manual(values=colors)+theme_classic()




High_ML_rate=High_ML/(High_ML+High_HC)
High_HC_rate=High_HC/(High_ML+High_HC)



Low_ML_rate=Low_ML/(Low_ML+Low_HC)
Low_HC_rate=Low_HC/(Low_ML+Low_HC)


Heterogeneity_ML_rate=Heterogeneity_ML/(Heterogeneity_ML+Heterogeneity_HC)
Heterogeneity_HC_rate=Heterogeneity_HC/(Heterogeneity_ML+Heterogeneity_HC)



confusion_ = data.frame(
  Group = c("High Risk", "Low Risk","Heterogeneity"),
  ML = c(High_ML_rate,Low_ML_rate,Heterogeneity_ML_rate),
  HC = c(High_HC_rate, Low_HC_rate,Heterogeneity_HC_rate)
)

confusion_dfm = melt(confusion_, id.vars=c("Group"), measure.vars=c("ML","HC"),
                     variable.name="Response", value.name="Frequency")
confusion_dfm$Group= factor(confusion_dfm$Group, levels = confusion_$Group)
confusion_dfm$Frequency=round(confusion_dfm$Frequency, digits = 3)
colors<-rev(c('#d5ead9','#f5dfc6','#65d6c6' ))
ggplot(confusion_dfm, aes(x = Group, y = Frequency, group = Response)) + 
  geom_col(aes(fill=Response)) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.25)) +
  geom_text(aes(label = Frequency), position = position_stack(vjust = .5), size = 3)+theme_bw()+ 
  theme(panel.grid=element_blank(),
        axis.text = element_text(size = 8,colour = 'black',family="Arial"),
        axis.title = element_text(size = 12,colour = 'black',family="Arial"))+
  scale_fill_manual(values=colors)+theme_classic()























rm(list = ls()) 
library(ggalluvial)
library(ggplot2)
library(cowplot)
library(dplyr)


inputFile="Shock_diagram.csv"  
outFile="diagram.pdf"    
setwd("Path") 
rt=read.table(inputFile, header = T, sep=",", check.names=F)[,c(3,2)]     
corLodes=to_lodes_form(rt, axes = 1:ncol(rt), id = "Cohort")


mycol <- rep(c("#df9b7e","#e1797b","#3d97ab","#7c67ab","#9eb6c9","#c7a32d","#B449B7"),7)
ggplot(corLodes, aes(x = x, stratum = stratum, alluvium = Cohort,fill = stratum, label = stratum)) +
  scale_x_discrete(expand = c(0, 0)) +  

  geom_flow(width = 2/10,aes.flow = "forward") + 
  geom_stratum(alpha = .9,width = 2/10) +

  geom_text(stat = "stratum", size = 2,color="black") +
  xlab("") + ylab("") + theme_bw() + 
  theme(axis.line = element_blank(),axis.ticks = element_blank(),axis.text.y = element_blank()) +
  theme(panel.grid =element_blank()) + 
  theme(panel.border = element_blank()) + 
  ggtitle("") + guides(fill = FALSE)                            



library(survival)
library(survminer)
KaplanMeier_Plot<-function(PD,Y_labels,titles){
  Test_Lung_PD0=PD
  Pre_Labels=Test_Lung_PD0$Pre_Label
  Test_Lung_PD0$Pre_Label<-factor(Pre_Labels,levels = c("Low_Risk","High_Risk","Heterogeneity"))
  fit<-survfit(Surv(OS,OS.State)~Pre_Label,data=Test_Lung_PD0)
  P0<-ggsurvplot(
    fit = fit,
    title=titles,
    risk.table = TRUE,
    risk.table.col="strata",
    ggtheme =   theme_classic(),
    palette=c("#536F9D","#C8825E","#8C8C8C"),
    pval = TRUE,#log-rank
    pval.method = TRUE,
    legend.labs=c("Low_Risk","High_Risk","Heterogeneity"),
    ylab=Y_labels,xlab = " Time (Years)",
    #tables.theme = theme_cleantable(),
    tables.height = 0.25,
    font.x = c(1, "plain", "white"),
    font.y = c(12, "plain", "black"),
    font.tickslab = c(8, "plain", "black"),
    xlim=c(0, 15),
    ylim = c(0, 1),
    size = 0.5,
    censor.size=6
  )
  return(P0)
}





##ANPY
inputFile=c('AFile.csv','NFile.csv','PFile.csv','YFile.csv')

setwd("Path") 
Data_A<-read.csv(inputFile[1])
Data_N<-read.csv(inputFile[2])
Data_P<-read.csv(inputFile[3])
Data_Y<-read.csv(inputFile[4])
Y_labels="Overall Survival(OS)"
DFS=1
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  Data_A$OS=Data_A$DFS
  Data_A$OS.State=Data_A$DFS.State
  Data_N$OS=Data_N$DFS
  Data_N$OS.State=Data_N$DFS.State
  Data_P$OS=Data_P$DFS
  Data_P$OS.State=Data_P$DFS.State
  Data_Y$OS=Data_Y$DFS
  Data_Y$OS.State=Data_Y$DFS.State
}

titles="Subtype A"
Pre_Label<-Data_A$Pre_Label
Test_Lung_PD0=Data_A
P1=KaplanMeier_Plot(Data_A,Y_labels,titles)

titles="Subtype N"
Pre_Label<-Data_N$Pre_Label
Test_Lung_PD0=Data_N
P2=KaplanMeier_Plot(Data_N,Y_labels,titles)
titles="Subtype P"
Pre_Label<-Data_P$Pre_Label

Test_Lung_PD0=Data_P
P3=KaplanMeier_Plot(Data_P,Y_labels,titles)
titles="Subtype Y"
Pre_Label<-Data_Y$Pre_Label

Test_Lung_PD0=Data_Y
P4=KaplanMeier_Plot(Data_Y,Y_labels,titles)

splots <- list()
splots[[1]] <- P1
splots[[2]] <- P2
splots[[3]] <- P3
splots[[4]] <- P4
res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)
setwd("Path")

ggsave(paste0("ANPY_DFS",".pdf"), plot =res,width = 25, height = 25, units = "cm")









##NE
setwd("Path")
File_name=c("NEFile.csv")
NE_DATA<- read.csv(File_name)
NE_High=NE_DATA[c(NE_DATA$NE_Type=="NE-High"),]
NE_Low=NE_DATA[c(NE_DATA$NE_Type=="NE-Low"),]


Y_labels="Overall Survival(OS)"
DFS=0
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  NE_High$OS=NE_High$Val_DFS
  NE_High$OS.State=NE_High$Val_DFState
  NE_Low$OS=NE_Low$Val_DFS
  NE_Low$OS.State=NE_Low$Val_DFState
  
}else{
  NE_High$OS=NE_High$Val_OS
  NE_High$OS.State=NE_High$Val_OSState
  NE_Low$OS=NE_Low$Val_OS
  NE_Low$OS.State=NE_Low$Val_OSState
}
Test_Lung_PD0=NE_High
titles="Subtype NE-High"
P1=KaplanMeier_Plot(NE_High,Y_labels,titles)
Test_Lung_PD0=NE_Low
titles="Subtype NE-Low"
P2=KaplanMeier_Plot(NE_Low,Y_labels,titles)

DFS=1
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  NE_High$OS=NE_High$Val_DFS
  NE_High$OS.State=NE_High$Val_DFState
  NE_Low$OS=NE_Low$Val_DFS
  NE_Low$OS.State=NE_Low$Val_DFState
  
}else{
  NE_High$OS=NE_High$Val_OS
  NE_High$OS.State=NE_High$Val_OSState
  NE_Low$OS=NE_Low$Val_OS
  NE_Low$OS.State=NE_Low$Val_OSState
}
Test_Lung_PD0=NE_High
titles="Subtype NE-High"
P3=KaplanMeier_Plot(NE_High,Y_labels,titles)
Test_Lung_PD0=NE_Low
titles="Subtype NE-Low"
P4=KaplanMeier_Plot(NE_Low,Y_labels,titles)


splots <- list()
splots[[1]] <- P1
splots[[3]] <- P2
splots[[2]] <- P3
splots[[4]] <- P4
res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)
setwd("Path")
ggsave(paste0("NE",".pdf"), plot =res,width = 25, height = 25, units = "cm")



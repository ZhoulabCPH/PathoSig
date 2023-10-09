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
library(survminer)

rm(list = ls()) 
KaplanMeier_Plot<-function(PD,Y_labels,titles){
  Test_Lung_PD0=PD
  Pre_Labels=Test_Lung_PD0$Pre_Label
  Test_Lung_PD0$Pre_Label<-factor(Pre_Labels,levels = c("Low_Risk","High_Risk","Heterogeneity"))
  fit<-survfit(Surv(OS,OSState)~Pre_Label,data=Test_Lung_PD0)
  P0<-ggsurvplot(
    fit = fit,
    title=titles,
    risk.table = TRUE,
    risk.table.col="strata",
    ggtheme =   theme_classic(),
    palette=c("#536F9D","#C8825E","#8C8C8C"),
    pval = TRUE,#log-rank检验
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

##Stage I-II and III-IV 
setwd("Path")
Discover<-read.csv('Discover.csv')
C_SCLC<-read.csv('C_SCLC.csv')
P_SCLC<-read.csv('P_SCLC.csv')

Y_labels="Overall Survival(OS)"
DFS=1
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  Discover$OS=Discover$DFS
  Discover$OSState=Discover$DFSState
  
  C_SCLC$OS=C_SCLC$DFS
  C_SCLC$OSState=C_SCLC$DFSState
  
  P_SCLC$OS=P_SCLC$DFS
  P_SCLC$OSState=P_SCLC$DFSState
  
}



##Stage-I-II
Stages1=1
Stages2=2

titles="P_SCLC(I-II)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Pre_Label<-Test_Lung_PD0$Pre_Label
P1<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


titles="C_SCLC(I-II)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Pre_Label<-Test_Lung_PD0$Pre_Label
P2<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

##Stage-III-IV
Stages1=3
Stages2=4

titles="P_SCLC(III-IV)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Pre_Label<-Test_Lung_PD0$Pre_Label
P3<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


titles="C_SCLC(III-IV)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Pre_Label<-Test_Lung_PD0$Pre_Label
P4<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


splots <- list()
splots[[1]] <- P1
splots[[2]] <- P2
splots[[3]] <- P3
splots[[4]] <- P4
res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)
setwd("Path")
ggsave(paste0(Y_labels,".pdf"), plot =res,width = 25, height = 25, units = "cm")



##################LymphaticMetastasis


setwd("Path")
Discover<-read.csv('Discover.csv')
C_SCLC<-read.csv('C_SCLC.csv')
P_SCLC<-read.csv('P_SCLC.csv')

Y_labels="Overall Survival(OS)"
DFS=0
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  Discover$OS=Discover$DFS
  Discover$OSState=Discover$DFSState
  
  C_SCLC$OS=C_SCLC$DFS
  C_SCLC$OSState=C_SCLC$DFSState
  
  P_SCLC$OS=P_SCLC$DFS
  P_SCLC$OSState=P_SCLC$DFSState
  
}


titles="P_SCLC(N 0)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$LymphaticMetastasis==0,]
Pre_Label<-Test_Lung_PD0$Pre_Label

P1<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

titles="P_SCLC(N 1)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$LymphaticMetastasis==1,]
Pre_Label<-Test_Lung_PD0$Pre_Label

P2<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

titles="C_SCLC(N 0)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$LymphaticMetastasis==0,]
Pre_Label<-Test_Lung_PD0$Pre_Label
P3<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


titles="C_SCLC(N 1-3)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$LymphaticMetastasis==1,]
Pre_Label<-Test_Lung_PD0$Pre_Label

P4<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

splots <- list()
splots[[1]] <- P1
splots[[3]] <- P3
splots[[4]] <- P4
splots[[2]] <- P2
res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)

setwd("Path")
ggsave(paste0(Y_labels,".pdf"), plot =res,width = 25, height = 25, units = "cm")




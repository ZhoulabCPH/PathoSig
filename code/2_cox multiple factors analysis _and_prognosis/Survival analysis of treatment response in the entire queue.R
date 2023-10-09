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
  fit<-survfit(Surv(Val_OS,Val_OSState)~Pre_Label,data=Test_Lung_PD0)
  P0<-ggsurvplot(
    fit = fit,
    title=titles,
    risk.table = TRUE,
    risk.table.col="strata",
    ggtheme =   theme_classic(),
    palette=c("#536F9D","#C8825E","#8C8C8C"), 
    pval = TRUE,
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


setwd("Path")
Discover<- read.csv('Discover_Traditional.csv')
Val_2<- read.csv('External_1_Traditional.csv')
Val_1<- read.csv('External_2_Traditional.csv')
Neoadjuvant_T<-read.csv('Neoadjuvant_T.csv')


Y_labels="Overall Survival(OS)"
DFS=1
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  Discover$Val_OS=Discover$Val_DFS
  Discover$Val_OSState=Discover$Val_DFState
  
  Val_2$Val_OS=Val_2$Val_DFS
  Val_2$Val_OSState=Val_2$Val_DFState

  Val_1$Val_OS=Val_1$Val_DFS
  Val_1$Val_OSState=Val_1$Val_DFState
  
  Neoadjuvant_T$Val_OS=Neoadjuvant_T$Val_DFS
  Neoadjuvant_T$Val_OSState=Neoadjuvant_T$Val_DFState
  
}



Pre_Label<-Neoadjuvant_T$Pre_Label
Neoadjuvant_T$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk","Heterogeneity"))
titles="Neoadjuvant_T"
Test_Lung_PD0=Neoadjuvant_T
P0<-KaplanMeier_Plot(Neoadjuvant_T,Y_labels,titles)




#1.Discovery
Pre_Label<-Discover$Pre_Label
Discover$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk","Heterogeneity"))
titles="Discover"
Test_Lung_PD0=Discover
P1<-KaplanMeier_Plot(Discover,Y_labels,titles)


#2.Val-1(P-SCLC)
Pre_Label<-Val_2$Pre_Label
Val_2$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk","Heterogeneity"))
titles="Val-1(P-SCLC)"
Test_Lung_PD0=Val_2
P2<-KaplanMeier_Plot(Val_2,Y_labels,titles)

#2.Val-2(C-SCLC)
Pre_Label<-Val_1$Pre_Label
Val_1$Pre_Label<-factor(Pre_Label,levels =  c("Low_Risk","High_Risk","Heterogeneity"))
titles="Val-2(C-SCLC)"
Test_Lung_PD0=Val_1
P3<-KaplanMeier_Plot(Val_1,Y_labels,titles)



splots <- list()
splots[[1]] <- P1
splots[[2]] <- P2
splots[[3]] <- P3
res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 3, nrow = 1)
setwd("Path")
ggsave(paste0(Y_labels,".pdf"), plot = res, width = 13.69, height = 5.27, units = "in", dpi = 300)



















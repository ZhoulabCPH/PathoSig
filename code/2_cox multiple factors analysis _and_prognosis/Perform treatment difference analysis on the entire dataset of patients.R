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


KaplanMeier_Plot<-function(PD,Y_labels,titles,type = "l"){
  
  fit<-survfit(Surv(Val_OS,Val_OSState)~Label,data=Test_Lung_PD0)
  Three_and_five_KM=summary(fit,time=c(3,5))
  P0<-ggsurvplot(
    fit = fit,
    title=titles,
    risk.table = TRUE,
    risk.table.col="strata",
    ggtheme =   theme_classic(),
    #palette=c("#536F9D","#C8825E"), 
    pval = TRUE,
    pval.method = TRUE,
    legend.labs=c("Traditional","Neoadjuvant"),
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
  res_cox<-coxph(Surv(Val_OS,Val_OSState)~Label,data=Test_Lung_PD0)
  P0$plot <-P0$plot+ 
    annotate("text",x = 3.5, y = 0.12,label = paste("HR :",round(summary(res_cox)$conf.int[1],4))) + 
    annotate("text",x = 3.5, y = 0.05,label = paste("(","95%CI:",round(summary(res_cox)$conf.int[3],4),"-",round(summary(res_cox)$conf.int[4],4),")",sep = ""))
  P0$table<-P0$table+ theme(
    plot.title = element_text(size=12),
    axis.text = element_text(size=8),
    legend.key = element_rect(fill = "white", colour = "black"),
  )+theme_classic(base_size = 12,
                  base_line_size = 0.5)
  
  return(P0)
}

setwd("Path")
All_Datasets<- read.csv('Thread.csv')

Y_labels="Overall Survival(OS)"
DFS=0
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  All_Datasets$Val_OS=All_Datasets$Val_DFS
  All_Datasets$Val_OSState=All_Datasets$Val_DFState

}


titles="All"
Test_Lung_PD0=All_Datasets
Label<-Test_Lung_PD0$TreatmentMode
Test_Lung_PD0$Label<-factor(Label,levels = c(0,1))
P0<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


titles="Low_Risk"
Test_Lung_PD0=All_Datasets
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$Pre_Label=="Low_Risk" ,]
Label<-Test_Lung_PD0$TreatmentMode
Test_Lung_PD0$Label<-factor(Label,levels = c(0,1))
P1<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

titles="High_Risk"
Test_Lung_PD0=All_Datasets
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$Pre_Label=="High_Risk" ,]
Label<-Test_Lung_PD0$TreatmentMode
Test_Lung_PD0$Label<-factor(Label,levels = c(0,1))
P2<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

titles="Heterogeneity"
Test_Lung_PD0=All_Datasets
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$Pre_Label=="Heterogeneity" ,]
Label<-Test_Lung_PD0$TreatmentMode
Test_Lung_PD0$Label<-factor(Label,levels = c(0,1))
P3<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


splots <- list()
splots[[1]] <- P1
splots[[2]] <- P2
splots[[3]] <- P3

res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 3, nrow = 1)





setwd("Path")
Discover_Different<- read.csv('Discover.csv')
External1_Different<- read.csv('External1.csv')
External2_Different<- read.csv('External2.csv')

Y_labels="Overall Survival(OS)"
DFS=0
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  
  Discover_Different$Val_OS=Discover_Different$Val_DFS
  Discover_Different$Val_OSState=Discover_Different$Val_DFState
  
  External1_Different$Val_OS=External1_Different$Val_DFS
  External1_Different$Val_OSState=External1_Different$Val_DFState
  
  External2_Different$Val_OS=External2_Different$Val_DFS
  External2_Different$Val_OSState=External2_Different$Val_DFState
  
}


############1.Discovery############
#1).High vs. Low
Label<-Discover_Different$Pre_Label
Discover_Different$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
titles="Discover"
Test_Lung_PD0=Discover_Different
P1<-KaplanMeier_Plot(Discover_Different,Y_labels,titles)

#2).Heterogeneity vs. Low
Label<-Discover_Different$Pre_Label
Discover_Different$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
titles="Discover"
Test_Lung_PD0=Discover_Different
P2<-KaplanMeier_Plot(Discover_Different,Y_labels,titles)


############1.Independent1############
#1).High vs. Low
Label<-External2_Different$Pre_Label
External2_Different$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
titles="Val-1(P-SLCL)"
Test_Lung_PD0=External2_Different
P3<-KaplanMeier_Plot(External2_Different,Y_labels,titles)

#2).Heterogeneity vs. Low
Label<-External2_Different$Pre_Label
External2_Different$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
titles="Val-1(P-SLCL)"
Test_Lung_PD0=External2_Different
P4<-KaplanMeier_Plot(External2_Different,Y_labels,titles)


############1.Independent2############
#1).High vs. Low
Label<-External1_Different$Pre_Label
External1_Different$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
titles="Val-2(C-SLCL)"
Test_Lung_PD0=External1_Different
P5<-KaplanMeier_Plot(External1_Different,Y_labels,titles)

#2).Heterogeneity vs. Low
Label<-External1_Different$Pre_Label
External1_Different$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
titles="Val-2(C-SLCL)"
Test_Lung_PD0=External1_Different
P6<-KaplanMeier_Plot(External1_Different,Y_labels,titles)



splots <- list()
splots[[1]] <- P1
splots[[2]] <- P2
splots[[3]] <- P3
splots[[4]] <- P4
splots[[5]] <- P5
splots[[6]] <- P6

res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 3, nrow = 2)




##########Stage

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


############1.P_SCLC############
#1).High vs. Low

Stages1=1
Stages2=2

titles="P_SCLC(I-II)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
P1<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

#2).Heterogeneity vs. Low

titles="P_SCLC(I-II)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
P2<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

Stages1=3
Stages2=4


titles="P_SCLC(III-IV)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
P3<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

#2).Heterogeneity vs. Low
titles="P_SCLC(I-II)"
Test_Lung_PD0=P_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
P4<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


splots <- list()
splots[[1]] <- P1
splots[[3]] <- P2
splots[[2]] <- P3
splots[[4]] <- P4

res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)





############2.C_SCLC############
#1).High vs. Low

Stages1=1
Stages2=2

titles="C_SCLC(I-II)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
P1<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

#2).Heterogeneity vs. Low

titles="C_SCLC(I-II)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
P2<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

Stages1=3
Stages2=4


titles="C_SCLC(III-IV)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","High_Risk"))
P3<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)

#2).Heterogeneity vs. Low
titles="C_SCLC(I-II)"
Test_Lung_PD0=C_SCLC
Test_Lung_PD0=Test_Lung_PD0[Test_Lung_PD0$AJCCTNM==Stages1 | Test_Lung_PD0$AJCCTNM==Stages2,]
Label<-Test_Lung_PD0$Pre_Label
Test_Lung_PD0$Label<-factor(Label,levels = c("Low_Risk","Heterogeneity"))
P4<-KaplanMeier_Plot(Test_Lung_PD0,Y_labels,titles)


splots <- list()
splots[[1]] <- P1
splots[[3]] <- P2
splots[[2]] <- P3
splots[[4]] <- P4

res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 2, nrow = 2)






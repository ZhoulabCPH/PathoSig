library(dplyr)
library(survival)
library(survminer)
library(ggplot2)
library(cutoff)
library(ggpubr)
library(reshape2)
library(forestplot)
library(zCompositions)
library(plyr)
library(ggpubr)
library(tibble)
rm(list = ls()) 
KaplanMeier_Plot<-function(PD,Y_labels,titles,type = "l"){
  Test_Lung_PD0=PD
  fit<-survfit(Surv(Val_OS,Val_OSState)~Pre_Label,data=Test_Lung_PD0)
  #Three_and_five_KM=summary(fit,time=c(3,5))
  P0<-ggsurvplot(
    fit = fit,
    title=titles,
    risk.table = TRUE,
    risk.table.col="strata",
    ggtheme =   theme_classic(),
    palette=c("#536F9D","#C8825E"), 
    pval = TRUE,
    pval.method = TRUE,
    legend.labs=c("Low_Risk","High_Risk"),
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
  #P2$plot<-P2$plot+ theme(panel.grid=element_blank())
  res_cox<-coxph(Surv(Val_OS,Val_OSState)~Pre_Label,data=Test_Lung_PD0)
  P0$plot <-P0$plot+ 
    annotate("text",x = 3.5, y = 0.12,label = paste("HR :",round(summary(res_cox)$conf.int[1],2))) + 
    annotate("text",x = 3.5, y = 0.05,label = paste("(","95%CI:",round(summary(res_cox)$conf.int[3],2),"-",round(summary(res_cox)$conf.int[4],2),")",sep = ""))
  P0$table<-P0$table+ theme(
    plot.title = element_text(size=12),
    axis.text = element_text(size=8),
    legend.key = element_rect(fill = "white", colour = "black"),
  )+theme_classic(base_size = 12,
                  base_line_size = 0.5)
  
  return(P0)
}



Discover_Path="Path"
setwd(Discover_Path)
Discover="Discover.csv"
Discover_Lung_PDS<- read.csv(Discover)


Y_labels="Overall Survival(OS)"
titles="Discover"

Pre_Label<-Discover_Lung_PDS$Pre_Label
Discover_Lung_PDS$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Discover_Lung_PDS
P1<-KaplanMeier_Plot(Discover_Lung_PDS,Y_labels,titles)










dataframe_PD=Discover_Lung_PDS[,3:54]
Drop0<-UniCoxID_Drop0(Discover_Lung_PDS,dataframe_PD)
Cluster_P1=Drop0[[1]]
new_train_pd=Drop0[[2]]
len_col = ncol(new_train_pd)
Train_Lung_PD_One=new_train_pd[,c(Cluster_P1,len_col-1,len_col)]##去除DFS的信息

Milt_P<-MiltyCox(Train_Lung_PD_One)
if(length(Milt_P)==2){
  print("No!")
  next
}


Train_Lung_PD_Two=Train_Lung_PD_One[,Milt_P]
Mulit_Plot(Train_Lung_PD_Two)

Cut_Info=Threshold_TestKM(Train_Lung_PD_Two)
Cutoff_median<-Cut_Info[[1]]
Cutoff_surv<-Cut_Info[[2]]
cox.mod<-Cut_Info[[3]]
Weight<-Cut_Info[[4]]
Cutoff=Cutoff_median



Y_labels="Overall Survival(OS)"

coef(cox.mod)

titles="Discover"
Pre<-predict(cox.mod,newdata = Discover_Lung_PDS)
Discover_Lung_PDS$Score=Pre
Discover_Lung_PDS$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Discover_Lung_PDS$Pre_Label
Discover_Lung_PDS$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Discover_Lung_PDS
P1<-KaplanMeier_Plot(Discover_Lung_PDS,Y_labels,titles)


#2.

titles="Independent1"
Pre<-predict(cox.mod,newdata = Independent_df1_Lung_PD)
Independent_df1_Lung_PD$Score=Pre
Independent_df1_Lung_PD$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Independent_df1_Lung_PD$Pre_Label
Independent_df1_Lung_PD$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Independent_df1_Lung_PD
P2<-KaplanMeier_Plot(Independent_df1_Lung_PD,Y_labels,titles)

#3.

titles="Independent2"
Pre<-predict(cox.mod,newdata = Independent_df2_Lung_PD)
Independent_df2_Lung_PD$Score=Pre
Independent_df2_Lung_PD$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Independent_df2_Lung_PD$Pre_Label
Independent_df2_Lung_PD$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Independent_df2_Lung_PD
P3<-KaplanMeier_Plot(Independent_df2_Lung_PD,Y_labels,titles)




DFS=1
if(DFS==1){
  Y_labels="Disease-free survival(DFS)"
  Discover_Lung_PDS$Val_OS=Discover_Lung_PDS$Val_DFS
  Discover_Lung_PDS$Val_OSState=Discover_Lung_PDS$Val_DFState
  
  Independent_df1_Lung_PD$Val_OS=Independent_df1_Lung_PD$Val_DFS
  Independent_df1_Lung_PD$Val_OSState=Independent_df1_Lung_PD$Val_DFState
  #Test_Lung_PD<- read.csv('Test00000009_UmapKM.csv')[,c(3:52,DFS_id)]
  Independent_df2_Lung_PD$Val_OS=Independent_df2_Lung_PD$Val_DFS
  Independent_df2_Lung_PD$Val_OSState=Independent_df2_Lung_PD$Val_DFState
}
titles="Discover"
Pre<-predict(cox.mod,newdata = Discover_Lung_PDS)
Discover_Lung_PDS$Score=Pre
Discover_Lung_PDS$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Discover_Lung_PDS$Pre_Label
Discover_Lung_PDS$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Discover_Lung_PDS
P4<-KaplanMeier_Plot(Discover_Lung_PDS,Y_labels,titles)


#2.

titles="Independent1"
Pre<-predict(cox.mod,newdata = Independent_df1_Lung_PD)
Independent_df1_Lung_PD$Score=Pre
Independent_df1_Lung_PD$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Independent_df1_Lung_PD$Pre_Label
Independent_df1_Lung_PD$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Independent_df1_Lung_PD
P5<-KaplanMeier_Plot(Independent_df1_Lung_PD,Y_labels,titles)

#3.

titles="Independent2"
Pre<-predict(cox.mod,newdata = Independent_df2_Lung_PD)
Independent_df2_Lung_PD$Score=Pre
Independent_df2_Lung_PD$Pre_Label<-ifelse(Pre<Cutoff,"Low_Risk","High_Risk")
Pre_Label<-Independent_df2_Lung_PD$Pre_Label
Independent_df2_Lung_PD$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","High_Risk"))
Test_Lung_PD0=Independent_df2_Lung_PD
P6<-KaplanMeier_Plot(Independent_df2_Lung_PD,Y_labels,titles)

splots <- list()
splots[[1]] <- P1
splots[[3]] <- P2
splots[[5]] <- P3
splots[[2]] <- P4
splots[[4]] <- P5
splots[[6]] <- P6

res<-arrange_ggsurvplots(splots, print = TRUE,
                         ncol = 3, nrow = 2)


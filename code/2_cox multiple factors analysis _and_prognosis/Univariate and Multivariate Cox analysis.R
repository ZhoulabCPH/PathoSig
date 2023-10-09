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

ForestTree<-function(){
  Cox_List<-data.frame(names(Data))
  Pre_Label<-Data[,c(Cox_List[6,1])]
  Data$Pre_Label<-factor(Pre_Label,levels = c("Low_Risk","Heterogeneity","High_Risk"))
  ##65 显著:Test_Lung_PD&ZC_Lung_PD
  Data$Age<-ifelse(Data$Age<=AgeTh,paste0("≤",AgeTh),paste0(">",AgeTh))
  Age<-Data[,c(Cox_List[8,1])]
  Data$Age<-factor(Age,levels = c(paste0("≤",AgeTh),paste0(">",AgeTh)))
  
  Data$AJCCTNM[Data$AJCCTNM==1]<-"I-II"
  Data$AJCCTNM[Data$AJCCTNM==2]<-"I-II"
  Data$AJCCTNM[Data$AJCCTNM==3]<-"III-IV"
  Data$AJCCTNM[Data$AJCCTNM==4]<-"III-IV"
  AJCCTNM<-Data[,c(Cox_List[10,1])]
  Data$AJCCTNM<-factor(AJCCTNM,levels = c("I-II","III-IV"))
  
  
  Data$Gender[Data$Gender==1]<-"M"
  Data$Gender[Data$Gender==2]<-"F"
  Gender<-Data[,c(Cox_List[7,1])]
  Data$Gender<-factor(Gender,levels = c("F","M"))
  
  Data$SmokingHistory[Data$SmokingHistory==1]<-"Yes"
  Data$SmokingHistory[Data$SmokingHistory==0]<-"No"
  SmokingHistory<-Data[,c(Cox_List[9,1])]
  Data$SmokingHistory<-factor(SmokingHistory,levels = c("No","Yes"))
  
  
  Data$LymphaticMetastasis[Data$LymphaticMetastasis==1]<-"Yes"
  Data$LymphaticMetastasis[Data$LymphaticMetastasis==0]<-"No"
  LymphaticMetastasis<-Data[,c(Cox_List[15,1])]
  Data$LymphaticMetastasis<-factor(LymphaticMetastasis,levels = c("No","Yes"))
  
  return(Data)
}

index=1

####OS cox
AgeTh=60
setwd("Path")
#Discover<- read.csv("Discover.csv")
File_Path=c('Discover.csv','External1.csv',
            'External2.csv')
savefilenames=c('Discover.pdf','Val-2(C-SCLC).pdf','Val-1(P-SCLC).pdf')


Data<- read.csv(File_Path[index])
Datas<-ForestTree()
covariates = names(Datas)[c(6,7,8,9,10)]
cox_data=Datas
univ_formulas  <- sapply(covariates,                
                         function(x) as.formula(paste('Surv(OS, OSState)~', x)))
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = cox_data)}) 

res = list()
for (i in noquote(names(univ_models))) {
  out_multi <- cbind(
    coef=summary(univ_models[[i]])$coefficients[,"coef"],
    HR= summary(univ_models[[i]])$conf.int[,"exp(coef)"],
    HR.95L=summary(univ_models[[i]])$conf.int[,"lower .95"],
    HR.95H=summary(univ_models[[i]])$conf.int[,"upper .95"],
    pvalue=summary(univ_models[[i]])$coefficients[,"Pr(>|z|)"])
  res[[i]] = out_multi
}

res = rbind(res$Pre_Label,res$Gender,res$Age,res$SmokingHistory,res$AJCCTNM)
res=as.data.frame(res)
forest_table <- cbind(c("Gene", covariates),
                      c("HR (95% CI)", paste0(round(res$HR,3),"(",round(res$HR.95L,3),"-",round(res$HR.95H,3),")")),
                      c("P-value", round(res$pvalue,3)))

csize <- data.frame(mean=c(NA, as.numeric(res$HR)),
                    lower=c(NA, as.numeric(res$HR.95L)),
                    upper=c(NA, as.numeric(res$HR.95H)))
setwd("Path")
pdf(savefilenames[index])
forestplot(labeltext = forest_table, 
           csize,
           graph.pos = 3,
           graphwidth = unit(5, "cm"),
           zero = 1,
           clip =c(0, 2),                      
           cex = 0.9,
           lineheight = "auto",
           boxsize = 0.05,
           fn.ci_norm = fpDrawNormalCI,
           lwd.ci = 1,
           ci.vertices = TRUE,
           lwd.xaxis = 1,
           ci.vertices.height = 0.08,
           col = fpColors(box = "skyblue", 
                          line = "darkblue"))
dev.off() 




###多cox
AgeTh=60
setwd("Path")
File_Path=c('Discover.csv','External1.csv',
            'External2.csv')
savefilenames=c('Discover.pdf','Val-2(C-SCLC).pdf','Val-1(P-SCLC).pdf')

Data<- read.csv(File_Path[index])
Datas<-ForestTree()
cox_data=Datas


cox.mod = coxph(Surv(OS,OSState) ~Pre_Label+Gender+Age+
                  SmokingHistory+AJCCTNM,data = cox_data)     
multiCoxSum = summary(cox.mod)

res2 <- cbind(as.data.frame(signif(multiCoxSum$coefficients[,"coef"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"exp(coef)"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"lower .95"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"upper .95"], digits=4)),
              as.data.frame(signif(multiCoxSum$coefficients[,"Pr(>|z|)"], digits=4)))
names(res2) = c("coef","HR","HR.95L","HR.95H","pvalue")

forest_table <- cbind(c("Gene", rownames(res2)),
                      c("HR (95% CI)", paste0(res2$HR,"(",res2$HR.95L,"-",res2$HR.95H,")")),
                      c("P-value", res2$pvalue))

csize <- data.frame(mean=c(NA, as.numeric(res2$HR)),
                    lower=c(NA, as.numeric(res2$HR.95L)),
                    upper=c(NA, as.numeric(res2$HR.95H)))


setwd("Path")
pdf(savefilenames[index])
forestplot(labeltext = forest_table, 
           csize,
           graph.pos = 3,
           graphwidth = unit(5, "cm"),
           zero = 1,
           clip =c(0, 2),                       
           cex = 0.9,
           lineheight = "auto",
           boxsize = 0.05,
           fn.ci_norm = fpDrawNormalCI,
           lwd.ci = 1,
           ci.vertices = TRUE,
           lwd.xaxis = 1,
           ci.vertices.height = 0.08,
           col = fpColors(box = "skyblue", 
                          line = "darkblue"))
dev.off() 























############DFS
setwd("Path")

AgeTh=60
File_Path=c('Discover_DFS.csv','External1_DFS.csv',
            'External2_DFS.csv',"Neoadjuvant_DFS.csv")
savefilenames=c('Discover_DFS.pdf','Val-2(C-SCLC)_DFS.pdf','Val-1(P-SCLC)_DFS.pdf',"Neoadjuvant_DFS.pdf")
index=4

Data<- read.csv(File_Path[index])
Datas<-ForestTree()

covariates = names(Datas)[c(6,7,8,9,10)]
cox_data=Datas
cox_data$OS=cox_data$DFS
cox_data$OSState=cox_data$DFSState

univ_formulas  <- sapply(covariates,                
                         function(x) as.formula(paste('Surv(OS, OSState)~', x)))
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = cox_data)}) 

res = list()
for (i in noquote(names(univ_models))) {
  out_multi <- cbind(
    coef=summary(univ_models[[i]])$coefficients[,"coef"],
    HR= summary(univ_models[[i]])$conf.int[,"exp(coef)"],
    HR.95L=summary(univ_models[[i]])$conf.int[,"lower .95"],
    HR.95H=summary(univ_models[[i]])$conf.int[,"upper .95"],
    pvalue=summary(univ_models[[i]])$coefficients[,"Pr(>|z|)"])
  res[[i]] = out_multi
}

res = rbind(res$Pre_Label,res$Gender,res$Age,res$SmokingHistory,res$AJCCTNM)
res=as.data.frame(res)
forest_table <- cbind(c("Gene", covariates),
                      c("HR (95% CI)", paste0(round(res$HR,3),"(",round(res$HR.95L,3),"-",round(res$HR.95H,3),")")),
                      c("P-value", round(res$pvalue,3)))

csize <- data.frame(mean=c(NA, as.numeric(res$HR)),
                    lower=c(NA, as.numeric(res$HR.95L)),
                    upper=c(NA, as.numeric(res$HR.95H)))
setwd("Path")
pdf(savefilenames[index])
forestplot(labeltext = forest_table, 
           csize,
           graph.pos = 3,
           graphwidth = unit(5, "cm"),
           zero = 1,
           clip =c(0, 2),                        # HR 范围
           cex = 0.9,
           lineheight = "auto",
           boxsize = 0.05,
           fn.ci_norm = fpDrawNormalCI,
           lwd.ci = 1,
           ci.vertices = TRUE,
           lwd.xaxis = 1,
           ci.vertices.height = 0.08,
           col = fpColors(box = "skyblue", 
                          line = "darkblue"))
dev.off() 


AgeTh=60
setwd("Path")


File_Path=c('Discover_DFS.csv','External1_DFS.csv',
            'External2_DFS.csv',"Neoadjuvant_DFS.csv")
savefilenames=c('Discover_DFS.pdf','Val-2(C-SCLC)_DFS.pdf','Val-1(P-SCLC)_DFS.pdf',"Neoadjuvant_DFS.pdf")

Data<- read.csv(File_Path[index])
Datas<-ForestTree()
cox_data=Datas
cox_data$OS=cox_data$DFS
cox_data$OSState=cox_data$DFSState

cox.mod = coxph(Surv(OS,OSState) ~Pre_Label+Gender+Age+
                  SmokingHistory+AJCCTNM,data = cox_data)  
multiCoxSum = summary(cox.mod)

res2 <- cbind(as.data.frame(signif(multiCoxSum$coefficients[,"coef"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"exp(coef)"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"lower .95"], digits=4)),
              as.data.frame(signif(multiCoxSum$conf.int[,"upper .95"], digits=4)),
              as.data.frame(signif(multiCoxSum$coefficients[,"Pr(>|z|)"], digits=4)))
names(res2) = c("coef","HR","HR.95L","HR.95H","pvalue")

forest_table <- cbind(c("Gene", rownames(res2)),
                      c("HR (95% CI)", paste0(res2$HR,"(",res2$HR.95L,"-",res2$HR.95H,")")),
                      c("P-value", res2$pvalue))

csize <- data.frame(mean=c(NA, as.numeric(res2$HR)),
                    lower=c(NA, as.numeric(res2$HR.95L)),
                    upper=c(NA, as.numeric(res2$HR.95H)))

setwd("Path")
pdf(savefilenames[index])
forestplot(labeltext = forest_table, 
           csize,
           graph.pos = 3,
           graphwidth = unit(5, "cm"),
           zero = 0,
           clip =c(0, 2),   #                  
           cex = 0.9,
           lineheight = "auto",
           boxsize = 0.05,
           fn.ci_norm = fpDrawNormalCI,
           ci.vertices.height = 0.08,
           col = fpColors(box = "skyblue", 
                          line = "darkblue"))

dev.off() 


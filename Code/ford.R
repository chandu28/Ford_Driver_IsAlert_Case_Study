
#working director
setwd('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\ford')
getwd()

#try caret
#data import
ford_train <- read.csv('fordTrain.csv',header = TRUE,sep = ',',na.strings = '')
ford_test <- read.csv('fordTest.csv',header = TRUE,sep = ',',na.strings = '')
str(ford_train)
head(ford_train)

#data univariate exploration
library(psych)
summary(ford_train)
describe(ford_train)

library(Hmisc)
describe(ford_train)

#determining factors dependent value
#bivariate analysis
#p1  and p4/P5/P6/P7 positive corelation
#P8 ALL 0 IGNORE V7 ALSO V9
cor(ford_train$P1,ford_train$P8)

#ENVIRONMENT
# e1 and e2/e5/e6/e7/e8/e10
cor(ford_train$E1,ford_train$E10)

#table for categorical #p8 neglect
#is alert cat
ford_train$IsAlert <- as.factor(ford_train$IsAlert)
class(ford_train$IsAlert)
#e3
ford_train$E3 <- as.factor(ford_train$E3)
class(ford_train$E3)
#E8
ford_train$E8 <- as.factor(ford_train$E8)
class(ford_train$E8)
#e9
#E7
ford_train$E7 <- as.factor(ford_train$E7)
class(ford_train$E7)
#E8
ford_train$E9 <- as.factor(ford_train$E9)
class(ford_train$E9)

#V5
ford_train$V5 <- as.factor(ford_train$V5)
class(ford_train$V5)

table(ford_train$E7)
#V10
ford_train$V10 <- as.factor(ford_train$V10)
class(ford_train$V5)

str(ford_train)

#histogram
#data transformation

if(!require(rcompanion))
{ install.packages("rcompanion")
  
}
#sknewness
if(!require(moments))
{ install.packages("moments")
}

library(moments)#skewness
library(rcompanion)


#histogram qq plot quantile plot skewness outliers
plotNormalHistogram(ford_train$P1)
qqnorm(ford_train$transform_p1,ylab= "frequency")
qqline(ford_train$transform_p1,col="red")
skewness(ford_train$P1)
out_p1<- NULL
out_p1 <- boxplot(ford_train$transform_p1)$out
out_p1
OutVals = boxplot(ford_train$P1, plot=FALSE)$out
(ford_train$P1%in%out_p1)

#p1 checking histogram
#checking quamtile plot
#checking skewness
plotNormalHistogram(ford_train$P1)
qqnorm(ford_train$P1,ylab= "frequency")
qqline(ford_train$P1,col="red")
# checking data there are negative values so a constant is added 
ford_train$transform_p1<- 50+ford_train$P1
ford_train$transform_p1_2 <- log(ford_train$transform_p1)
ford_train$I.D.C.V.1 <- sqrt(ford_train$transform_p1_2)
#skewness is less than 1
skewness(ford_train$I.D.C.V.1)
boxplot(ford_train$I.D.C.V.1)
plotNormalHistogram(ford_train$I.D.C.V.1)

# p2 same process is carried like before
plotNormalHistogram(ford_train$P2)
skewness(ford_train$P2)
ford_train$transform_p2 <- 50 + ford_train$P2
ford_train$transform_p2_1 <- sqrt(ford_train$transform_p2)
ford_train$I.D.C.V.2 <- ford_train$transform_p2_1
skewness(ford_train$I.D.C.V.2)
boxplot(ford_train$I.D.C.V.2)
plotNormalHistogram(ford_train$I.D.C.V.2)
#skewness is less than 1 so need no transformation

#p3

#some right skew was there
skewness(ford_train$P3)
#ford_train$transform_p3 <- log(ford_train$P3)
ford_train$I.D.C.V.3 <- (ford_train$P3)^(1/3)
skewness(ford_train$I.D.C.V.3)
plotNormalHistogram(ford_train$I.D.C.V.3)

#P4
plotNormalHistogram(ford_train$P4)
#SOME RIGHT SKEW
ford_train$transform_p4_1 <- sqrt(ford_train$P4)
ford_train$I.D.C.V.4 <- ford_train$transform_p4_1
plotNormalHistogram(ford_train$I.D.C.V.4)
boxplot(ford_train$I.D.C.V.4)
skewness(ford_train$I.D.C.V.4)

#p5
describe(ford_train)
plotNormalHistogram(ford_train$P5)
skewness(ford_train$P5)
boxplot(ford_train$P5)
ford_train$transform_p5_1 <- (1/(ford_train$P5))
skewness(ford_train$transform_p5_1)
plotNormalHistogram(ford_train$transform_p5_1)
ford_train$I.D.C.V.5 <- ford_train$transform_p5_1
plotNormalHistogram(ford_train$I.D.C.V.5)
boxplot(ford_train$I.D.C.V.5)
skewness(ford_train$I.D.C.V.5)


#p6

plotNormalHistogram(ford_train$P6)
skewness(ford_train$P6)
ford_train$transform_p6_1 <- log(ford_train$P6)
skewness(ford_train$transform_p6_1)
ford_train$transform_p6_2 <- 1/(ford_train$transform_p6_1)
ford_train$I.D.C.V.6 <- ford_train$transform_p6_2
skewness(ford_train$I.D.C.V.6)
plotNormalHistogram(ford_train$I.D.C.V.6)
boxplot(ford_train$I.D.C.V.6)


#p7

skewness(ford_train$P7)
ford_train$transform_p7_1 <- sqrt(ford_train$P7)
skewness(ford_train$transform_p7_1)
ford_train$I.D.C.V.7 <- ford_train$transform_p7_1
plotNormalHistogram(ford_train$I.D.C.V.7)
skewness(ford_train$I.D.C.V.7)
boxplot(ford_train$I.D.C.V.7)

#p8 all zeros
max(ford_train$P8)
min(ford_train$P8)

#e1 

skewness(ford_train$E1)
ford_train$I.D.C.V.8 <- ford_train$E1
skewness(ford_train$I.D.C.V.8)
plotNormalHistogram(ford_train$I.D.C.V.8)

#e2

skewness(ford_train$E2)
ford_train$I.D.C.V.9 <- ford_train$E2
skewness(ford_train$I.D.C.V.9)

#e3
ford_train$I.D.C.F.1 <- ford_train$E3


#e4 
plotNormalHistogram(ford_train$E4)
out_e4 <- boxplot(ford_train$E4)$out
out_e4
median(ford_train$E4)
ford_train$transform_e4_1 <- (ford_train$E4)^2
skewness(ford_train$transform_e4_1)
ford_train$transform_e4_2 <-  log(10+ford_train$transform_e4_1)
ford_train$transform_e4_3 <-  log(ford_train$transform_e4_2)
skewness(ford_train$transform_e4_3)
plotNormalHistogram(ford_train$transform_e4_3)
boxplot(ford_train$transform_e4_3)
max(ford_train$transform_e4_3)
min(ford_train$transform_e4_3)
ford_train$I.D.C.V.10 <- ford_train$transform_e4_3
skewness(ford_train$I.D.C.V.10)
plotNormalHistogram(ford_train$I.D.C.V.10)
boxplot(ford_train$I.D.C.V.10)

#e5
skewness(ford_train$E5)
ford_train$I.D.C.V.11 <- ford_train$E5

#e6

skewness(ford_train$E6)
ford_train$I.D.C.V.12 <- ford_train$E6

#e7
ford_train$I.D.C.F.2 <- ford_train$E7

#e8
ford_train$I.D.C.F.3 <- ford_train$E8

#e9
ford_train$I.D.C.F.4 <- ford_train$E9

#e10

skewness(ford_train$E10)
plotNormalHistogram(ford_train$E10)
ford_train$I.D.C.F.5 <- ford_train$E10

#e11 not yet id.c.v13 only skew till 2.57
max(ford_train$E11)
min(ford_train$E11)
sum(ford_train$E11==0)
ford_train$impute_e11 <- ifelse(ford_train$E11==0,
                                 1.315265,
                                 ford_train$E11)
skewness(ford_train$impute_e11)
plotNormalHistogram(ford_train$E11)
ford_train$transform_e11_1 <- log(ford_train$impute_e11)
skewness(ford_train$transform_e11_1)
plotNormalHistogram(ford_train$transform_e11_1)
ford_train$transform_e11_2 <- log1p(ford_train$transform_e11_1)
skewness(ford_train$transform_e11_2)
ford_train$transform_e11_3 <- (10+ford_train$transform_e11_2)^(1/2)
skewness(ford_train$transform_e11_3)
ford_train$transform_e11_4<- sqrt(ford_train$transform_e11_3)
skewness(ford_train$transform_e11_4)
ford_train$I.D.C.V.13<- ford_train$transform_e11_4
plotNormalHistogram(ford_train$I.D.C.V.13)
skewness(ford_train$I.D.C.V.13)

#v1

skewness(ford_train$V1)
plotNormalHistogram(ford_train$V1)
ford_train$transform_v1_1 <- (ford_train$V1)^2
skewness(ford_train$transform_v1_1)
plotNormalHistogram(ford_train$transform_v1_1)
ford_train$transform_v1_2 <- (ford_train$transform_v1_1)/100
skewness(ford_train$transform_v1_2)
plotNormalHistogram(ford_train$transform_v1_2)
boxplot(ford_train$transform_v1_2)
ford_train$I.D.C.V.14 <- ford_train$transform_v1_2
skewness(ford_train$I.D.C.V.14)
plotNormalHistogram(ford_train$I.D.C.V.14)



#v2

plotNormalHistogram(ford_train$V2)
skewness(ford_train$V2)
ford_train$I.D.C.V.15 <- ford_train$V2
skewness(ford_train$I.D.C.V.15)
plotNormalHistogram(ford_train$I.D.C.V.15)


#v3

skewness(ford_train$V3)
plotNormalHistogram(ford_train$V3)
ford_train$I.D.C.V.16 <- ford_train$V3
skewness(ford_train$I.D.C.V.16)
plotNormalHistogram(ford_train$I.D.C.V.16)

#i.d.c.v 17
#v4 
max(ford_train$V4)
min(ford_train$V4)
median(ford_train$V4)
sum(ford_train$V4==0)
skewness(ford_train$V4)
plotNormalHistogram(ford_train$V4)
ford_train$transform_v4_1 <- log1p(ford_train$V4)
skewness(ford_train$transform_v4_1)
min(ford_train$transform_v4_1)
plotNormalHistogram(ford_train$transform_v4_1)
ford_train$transform_v4_2 <- log1p(ford_train$transform_v4_1)
skewness(ford_train$transform_v4_2)
plotNormalHistogram(ford_train$transform_v4_2)
boxplot(ford_train$transform_v4_2)
ford_train$I.D.C.V.17 <- ford_train$transform_v4_2
skewness(ford_train$I.D.C.V.17)
plotNormalHistogram(ford_train$I.D.C.V.17)
boxplot(ford_train$I.D.C.V.17)

#v5 
ford_train$I.D.C.F.5 <- ford_train$V5

#v6
skewness(ford_train$V6)
ford_train$I.D.C.V.18 <- ford_train$V6
skewness(ford_train$I.D.C.V.18)
plotNormalHistogram(ford_train$I.D.C.V.18)
boxplot(ford_train$I.D.C.V.18)


#v7 neglect all 0
max(ford_train$V6)
min(ford_train$V6)


#v8

skewness(ford_train$V8)
plotNormalHistogram(ford_train$V8)
ford_train$I.D.C.V.19 <- ford_train$V8
boxplot(ford_train$I.D.C.V.19)
skewness(ford_train$I.D.C.V.19)

#v9 all 0
max(ford_train$V9)
min(ford_train$V9)


#v10
ford_train$I.D.C.F.6 <- ford_train$V10

#v11
max(ford_train$V11)
min(ford_train$V11)
skewness(ford_train$V11)
plotNormalHistogram(ford_train$V11)
ford_train$transform_v11_1 <- log(ford_train$V11)
skewness(ford_train$transform_v11_1)
plotNormalHistogram(ford_train$transform_v11_1)
boxplot(ford_train$transform_v11_1)
ford_train$I.D.C.V.20 <- ford_train$transform_v11_1
skewness(ford_train$I.D.C.V.20)
plotNormalHistogram(ford_train$I.D.C.V.20)
boxplot(ford_train$I.D.C.V.20)


str(ford_train)


#  26 DEPDENT VARIABLE
# 6 FACTOR 20 CONTINOUS
# 3 ALL ZEROS
# I DEPENDENT
#checking skewness
#factor no skewness
#continous independent variable
skewness(ford_train$I.D.C.V.1)
skewness(ford_train$I.D.C.V.2)
skewness(ford_train$I.D.C.V.3)
skewness(ford_train$I.D.C.V.4)
skewness(ford_train$I.D.C.V.5) 
skewness(ford_train$I.D.C.V.6)
skewness(ford_train$I.D.C.V.7)
skewness(ford_train$I.D.C.V.8)
skewness(ford_train$I.D.C.V.9)
skewness(ford_train$I.D.C.V.10)
skewness(ford_train$I.D.C.V.11)
skewness(ford_train$I.D.C.V.12)
skewness(ford_train$I.D.C.V.13)
skewness(ford_train$I.D.C.V.14)
skewness(ford_train$I.D.C.V.15)
skewness(ford_train$I.D.C.V.16)
skewness(ford_train$I.D.C.V.17)
skewness(ford_train$I.D.C.V.18)
skewness(ford_train$I.D.C.V.19)
skewness(ford_train$I.D.C.V.20)

#dependent
ford_train$D.V <- ford_train$IsAlert


# creating data frame with d.v i.d.v factors
ford_dataframe <- NULL
ford_dataframe <- data.frame(ford_train$D.V,ford_train$I.D.C.F.1,ford_train$I.D.C.F.2,
                             ford_train$I.D.C.F.3,ford_train$I.D.C.F.4,ford_train$I.D.C.F.5,
                             ford_train$I.D.C.F.6,ford_train$I.D.C.V.1,ford_train$I.D.C.V.2,
                             ford_train$I.D.C.V.3,ford_train$I.D.C.V.4,ford_train$I.D.C.V.5,
                             ford_train$I.D.C.V.6,ford_train$I.D.C.V.7,ford_train$I.D.C.V.8,
                             ford_train$I.D.C.V.9,ford_train$I.D.C.V.10,ford_train$I.D.C.V.11,
                             ford_train$I.D.C.V.12,ford_train$I.D.C.V.13,ford_train$I.D.C.V.14,
                             ford_train$I.D.C.V.15,ford_train$I.D.C.V.16,ford_train$I.D.C.V.17,
                             ford_train$I.D.C.V.18,ford_train$I.D.C.V.19,ford_train$I.D.C.V.20)

#coreelation 0 no correaltion continous variable
# corelation 1 -1 positive and negative
rcorr(as.matrix(ford_dataframe))


#model bulding 

if(!require(caret))
{ install.packages("caret")
  
}
if(!require(dplyr))
{ install.packages("dplyr")
  
}
library(dplyr)
library(caret)
#glm
model_logistic <- train(ford_train.D.V ~ ford_train.I.D.C.F.1+ford_train.I.D.C.F.2+
                          ford_train.I.D.C.F.3+ford_train.I.D.C.F.4+ford_train.I.D.C.F.5+
                          ford_train.I.D.C.F.6+ford_train.I.D.C.V.1+ford_train.I.D.C.V.2+
                          ford_train.I.D.C.V.3+ford_train.I.D.C.V.4+ford_train.I.D.C.V.5+
                          ford_train.I.D.C.V.6+ford_train.I.D.C.V.7+ford_train.I.D.C.V.8+
                          ford_train.I.D.C.V.9+ford_train.I.D.C.V.10+ford_train.I.D.C.V.11+
                          ford_train.I.D.C.V.12+ford_train.I.D.C.V.13+ford_train.I.D.C.V.14+
                          ford_train.I.D.C.V.15+ford_train.I.D.C.V.16+ford_train.I.D.C.V.17+
                          ford_train.I.D.C.V.18+ford_train.I.D.C.V.19+ford_train.I.D.C.V.20,
                        data= ford_dataframe, method="glm",family="binomial")
                    

summary(model_logistic)
ford_dataframe$logistic_op <- predict.train(model_logistic,ford_dataframe,type = "raw")
table(ford_dataframe$logistic_op)
#confusion matrix
cm_glm <- table(ford_dataframe$logistic_op,ford_dataframe$ford_train.D.V)
cm_glm
accuracy_glm <- ((192251+313417)/604329 *100)
accuracy_glm


#pcm
library(caret)
predictors <- NULL
predicted <- train[c("IsAlert")]
predictors <- train(c("P1","P2","P3","P4","P5","P6","P7","E6","E7","E8","E9","E10","E11","V3","V4","V5","V6","V8","E4","E5","V1","V2","V10","V11"))
predicted <- train[c("IsAlert")]
predictors
pcm_model <- prcomp(predictors, scale. = T)

# decision tree
model_decison_tree <-train(ford_train.D.V ~ ford_train.I.D.C.F.1+ford_train.I.D.C.F.2+
                             ford_train.I.D.C.F.3+ford_train.I.D.C.F.4+ford_train.I.D.C.F.5+
                             ford_train.I.D.C.F.6+ford_train.I.D.C.V.1+ford_train.I.D.C.V.2+
                             ford_train.I.D.C.V.3+ford_train.I.D.C.V.4+ford_train.I.D.C.V.5+
                             ford_train.I.D.C.V.6+ford_train.I.D.C.V.7+ford_train.I.D.C.V.8+
                             ford_train.I.D.C.V.9+ford_train.I.D.C.V.10+ford_train.I.D.C.V.11+
                             ford_train.I.D.C.V.12+ford_train.I.D.C.V.13+ford_train.I.D.C.V.14+
                             ford_train.I.D.C.V.15+ford_train.I.D.C.V.16+ford_train.I.D.C.V.17+
                             ford_train.I.D.C.V.18+ford_train.I.D.C.V.19+ford_train.I.D.C.V.20,
                           data= ford_dataframe, method="ctree")
png(file = "ford_tree.png")
set.seed(1234)
summary(model_decison_tree)
model_decison_tree
plot(model_decison_tree)
dev.off()
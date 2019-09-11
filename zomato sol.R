#### loading required packages ####

library(data.table)
library(xgboost)
library(dplyr)
library(ggplot2)
library(caret)
library(MASS)
library(car)

#### loading data set ####

dat <- fread("zomato.csv")

#### data prep ####

dat <- dat[,!c("url","address","phone","listed_in(type)","menu_item","reviews_list","listed_in(city)","dish_liked")]
dat$rate <- substr(dat$rate,1,3)
dat$rate <- as.numeric(dat$rate)
dat <- na.omit(dat)
dat$cost <- dat$`approx_cost(for two people)`
dat$`approx_cost(for two people)` <- NULL
dat$cost<- as.numeric(gsub(",","",dat$cost))
dat$online_order <- as.factor(dat$online_order)
dat$location <- as.factor(dat$location)
dat <- dat %>% filter(votes >= 20)
dat <- na.omit(dat)
#### linear model #####

#split into train and test 

rt <- createDataPartition(dat$rate,times=1,p=0.7)
train <- dat[rt$Resample1,c("rate","online_order","cost","location")]
test <- dat[-rt$Resample1,c("online_order","cost","location")]

#fitting the model 

model <- lm(rate~online_order+cost+location,train)

anova(model)
plot(model)

#Large tail in Normal Q-Q Plot. Let us transform the predictor variable. 
m <- powerTransform(model)
lambda <- as.numeric(m$lambda)
model <- lm(rate^lambda~online_order+cost+location,train) 
summary(model)
anova(model)
plot(model)

#predictions

pred <- round(predict(model,test)^(1/lambda),1)

#forecast accuracy

accuracy <- 1- sum(abs(dat$rate[-c(rt$Resample1)] - pred))/sum(dat$rate[-c(rt$Resample1)])
final = cbind(dat$rate[-c(rt$Resample1)],pred)
colnames(final) <- c("actual","predicted")

final <- as.data.frame(final)

###### XGBoost ######

#hot encoding for categorical variables

dmy <- dummyVars(" ~ .", data = dat[,c("rate","online_order","cost","location")])
dat_1 <- data.frame(predict(dmy, newdata = dat[,c("rate","online_order","cost","location")]))

#split into train and test data

rt <- createDataPartition(dat_1$rate,times=1,p=0.7)
train <- dat_1[rt$Resample1,]
test <- dat_1[-rt$Resample1,]

train <- as.matrix(train)

#fitting the model

fit <- xgboost(data = train, label = train[,1],nrounds= 50, max.depth=6,objective="reg:linear")
pred <- round(predict(fit, newdata=as.matrix(test)),1)

#forecast accuracy
accuracy <- 1- sum(abs(dat$rate[-c(rt$Resample1)] - pred))/sum(dat$rate[-c(rt$Resample1)])

final = cbind(dat$rate[-c(rt$Resample1)],pred)
colnames(final) <- c("actual","predicted")

final <- as.data.frame(final)

library(readxl)
library(glmnet)
library(ggplot2)
library(caret)
library(Metrics)
library(plyr)
library(dplyr)
library(moments)
library(elasticnet)
library(knitr)

dataset <- read_excel("***.xlsx")
dataset[dataset==999]<-NA


# Of note: microstates of 0 value were put to NA from previous step of processing -> set back to 0
naindx<-which(is.na(dataset$FAgfp1))
dataset$FAgfp1[naindx]<-0
naindx<-which(is.na(dataset$FAgfp2))
dataset$FAgfp2[naindx]<-0
naindx<-which(is.na(dataset$FAgfp4))
dataset$FAgfp4[naindx]<-0

naindx<-which(is.na(dataset$FDgfp1))
dataset$FDgfp1[naindx]<-0
naindx<-which(is.na(dataset$FDgfp2))
dataset$FDgfp2[naindx]<-0
naindx<-which(is.na(dataset$FDgfp4))
dataset$FDgfp4[naindx]<-0

naindx<-which(is.na(dataset$Ngfp1))
dataset$Ngfp1[naindx]<-0
naindx<-which(is.na(dataset$Ngfp2))
dataset$Ngfp2[naindx]<-0
naindx<-which(is.na(dataset$Ngfp4))
dataset$Ngfp4[naindx]<-0

# select complete output measure of interest
na_vabs_indx<-which(is.na(dataset$VABSsoc))
data <- dataset[-na_vabs_indx, ]

demog_naindx<-which(is.na(data$elc8))
data <- data[-demog_naindx, ]

# scale variable of interest
Y<-(as.numeric(data$VABSsoc))
Y<-scale(Y^4)

# STRATIFIED PARTITION on CLINICAL OUTCOME for CROSS-VALIDATION

# repeated hold-out
#folds<-1
#partition<-createDataPartition(data$outcome,p=0.7,times=folds,list=FALSE)
#nresampling<-folds

#LOO-CV
partition<-groupKFold(data$id,k=nrow(data))
partition<-as.data.frame(partition)

#stratified 10-fold
#partition<-createFolds(factor(data$outcome), k = 10, list = FALSE)


nresampling<-nrow(data)


# INITIALIZE RESULTS VECTORS
best_lambdas<-vector()
cv_coeff<-vector()
cv_rmse<-vector()
prediction_ci<-vector()
prediction_shuffle<-vector()
prediction<-vector()
actual<-vector()
res<-vector()
fitted<-vector()

# FLAGS 
compute_predictionCI<-0
model_alpha<-0.1 #elnet

for(ii in 1:nresampling){
  
 partition_indx<-partition[,ii]
  #partition_indx<-which(partition!=ii)
  train<-data[partition_indx,]
  test<-data[-partition_indx,]
  
  # train and test sets
  trainfeat<-cbind(train[,3],train[,7:26])
  testfeat<-cbind(test[,3],test[,7:26])
  
  # combine train and test data for preprocessing
  all_data <- rbind(trainfeat,testfeat)
  
  others_naindx<-which(is.na(all_data),arr.ind=TRUE)
  all_data[others_naindx]<-0
  
  # determine skew for each numeric feature
  skewed_feats <- sapply(names(all_data),function(x){skewness(all_data[[x]],na.rm=TRUE)})
  
  # keep only features that exceed a threshold for skewness
  skewed_feats <- skewed_feats[abs(skewed_feats) > 0.7]
  
  to_transform<-names(skewed_feats)
  
  
  # transform excessively skewed features with sqrt
  for(x in to_transform) {
    all_data[[x]] <- sqrt(all_data[[x]])
  }
  
  
  # IF NEEDED scale data
  #names_data<-names(all_data)
  #all_data<-matrix(as.numeric(unlist(all_data)),nrow=nrow(all_data))
  #all_data<-scale(all_data)
  #all_data<-as.data.frame(all_data)
  #colnames(all_data)<-names_data
  #Y<-scale(Y)
  # to_scale<-names(all_data)
  # binary<-c(2)
  # to_scale<-to_scale[-binary]
  #all_data<-sapply(names(all_data),function(x){(all_data[[x]]-min(all_data[[x]]))/(max(all_data[[x]])-min(all_data[[x]]))})
  #Y<-(Y-min(Y))/(max(Y)-min(Y))
  
  # create data for training and test
  X_train <- all_data[1:nrow(train),]
  X_test <- all_data[(nrow(train)+1):nrow(all_data),]
  y <- Y[partition_indx]
  y_test <- Y[-partition_indx]
  
  # set up caret model training parameters
  # model specific training parameter
  CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",number=10,repeats=10,verboseIter=FALSE)
  
  # TEST OUT THE REGULARIZED REGRESSION MODEL [alpha=0 RIDGE; 0<alpha<1 ELASTIC NET; alpha=1 LASSO]
  
  #parameters
  lambdas <- c(seq(1,0,-0.001),0.00075,0.0005,0.0001)
  
  # train model
  set.seed(123)  # for reproducibility
  cat("Training the ELNET-regression model with LOO-CV...execution n",ii,"over ",nresampling,"\n")
  model <- train(x=X_train,y=y,method="glmnet",metric="RMSE",maximize=FALSE,preProc=c('scale','center'),trControl=CARET.TRAIN.CTRL,tuneGrid=expand.grid(alpha=model_alpha,lambda=lambdas))
  
  ggplot(data=filter(model$result,RMSE<1)) + geom_line(aes(x=lambda,y=RMSE))
  
  model
  
  avg_RMSE<-mean(model$resample$RMSE)
  avg_RMSE
  cv_rmse<-c(cv_rmse,avg_RMSE)
  
  best_lambdas<-c(best_lambdas,model$bestTune$lambda)
  
  # extract coefficients for the best performing model 
  coef <- data.frame(coef.name = dimnames(coef(model$finalModel,s=model$bestTune$lambda))[[1]],coef.value = matrix(coef(model$finalModel,s=model$bestTune$lambda)))
  
  # exclude the (Intercept) term
  coef <- coef[-1,]
  
  cv_coeff<-rbind(cv_coeff,coef[,2])
  
  # print summary of model results
  picked_features <- nrow(filter(coef,coef.value!=0))
  not_picked_features <- nrow(filter(coef,coef.value==0))
  
  
    cat("ElNet picked",picked_features,"variables and eliminated the other",not_picked_features,"variables\n")
  
  # sort coefficients in ascending order
  coef <- arrange(coef,-coef.value)
  
  # extract the top 10 and bottom 10 features
  imp_coef <- rbind(head(coef,10),tail(coef,10))
  
  ggplot(imp_coef) + geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),stat="identity") +
    coord_flip() + ggtitle("Coefficents in the Model") +
    theme(axis.title=element_blank())
  
  #predict test data
  predicted<-predict(model,newdata=X_test)
  
  prediction<-c(prediction,predicted)
  actual<-c(actual,y_test)
  res<-rbind(res,residuals(model))
  fitted<-rbind(fitted,fitted(model))
  
  if(compute_predictionCI==1){
    # BOOTSTRAP 95%CI
    B <- 1000
    pred.boot <- numeric(B)
    for (i in 1:B) {
      # train model
      boot <- sample(seq(nrow(X_train),1),nrow(X_train),replace=TRUE)
      set.seed(123)  # for reproducibility
      model.b <- train(x=X_train[boot,],y=y[boot],method="glmnet",metric="RMSE",maximize=FALSE,preProc=c("center","scale"),trControl=CARET.TRAIN.CTRL,tuneGrid=expand.grid(alpha=model_alpha,lambda=lambdas))
      pred.boot[i] <- predict(model.b, X_test)
    }
    prediction_ci<-rbind(prediction_ci,quantile(pred.boot, c(0.025, 0.975)))
  }
  
  # SHUFFLE TEST for P-VALUE
  #pp <- 1000
  #pred.shuffle <- numeric(pp)
  #for (i in 1:pp) {
  #  cat("Doing permutation test...pp=",i,"over 1000 \n")
  #  # train model with random outcome
  #  shuffle <- sample(seq(nrow(X_train),1))
  #  set.seed(123)  # for reproducibility
  #  model.s <- train(x=X_train,y=y[shuffle],method="glmnet",metric="RMSE",maximize=FALSE,preProc=c("center","scale"),trControl=CARET.TRAIN.CTRL,tuneGrid=expand.grid(alpha=model_alpha,lambda=lambdas))
  #  pred.shuffle[i] <- predict(model.s, X_test)
  #}
  #prediction_shuffle<-rbind(prediction_shuffle,pred.shuffle)
  
}


#residual<-exp(prediction)-exp(actual)
residual<-(prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4)-(actual*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4)

#check normality of residuals
qqnorm(residual)
qqline(residual)

#metrics
error<-rmse((prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4),(actual*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4))/(max(as.numeric(data$VABSsoc))-min(as.numeric(data$VABSsoc)))
rmse<-rmse((prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4),(actual*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4))
mae<-sum(abs(residual))/length(residual)
relative_error<-((prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4))/((actual*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4))
cbind((prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4),(actual*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4))

# shuffle p-value
shufflep_rmse<-0
shufflep_mae<-0

#for(ss in 1:ncol(prediction_shuffle)){
for(ss in 1:100000){
  cat("Doing permutation test...repetition",ss,"over 1000 \n")
  shuffle <- sample(seq(length(actual),1))
  #residual_shuffle<-prediction_shuffle[,ss]-actual
  #rmse_shuffle<-rmse(prediction_shuffle[,ss],actual)
  residual_shuffle<-(prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4)-((actual[shuffle]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4)
  rmse_shuffle<-rmse((prediction*attr(Y,"scaled:scale")+attr(Y,"scaled:center"))^(1/4),((actual[shuffle]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4))
  mae_shuffle<-sum(abs(residual_shuffle))/length(residual_shuffle)
  if(rmse_shuffle<rmse){shufflep_rmse<-shufflep_rmse+1}
  if(mae_shuffle<mae){shufflep_mae<-shufflep_mae+1}
}
shufflep_rmse<-shufflep_rmse/100000
shufflep_mae<-shufflep_mae/100000

# BOOTSTRAP 95%CI
B <- 1000
rmse.b <- numeric(B)
mae.b<-numeric(B)
for (i in 1:B) {
  # train model
  cat("Doing bootstrap for CIs...repetition",i,"over 1000 \n")
  boot <- sample(seq(length(prediction),1),length(prediction),replace=TRUE)
  residual.b<-((prediction[boot]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4)-((actual[boot]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4)
  mae.b[i]<-sum(abs(residual.b))/length(residual.b)
  rmse.b[i] <- rmse(((prediction[boot]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4),((actual[boot]*attr(Y,"scaled:scale")+attr(Y,"scaled:center")))^(1/4))
}
mae_ci<-quantile(mae.b, c(0.025, 0.975))
rmse_ci<-quantile(rmse.b, c(0.025, 0.975))

final_coeff<-colMeans(cv_coeff)
final_coeff_sd<-apply(cv_coeff,2,sd)
selected_coef <- data.frame(coef.name = names(trainfeat),coef.value = final_coeff)

nonzeroindx<-numeric()
for(nn in 1:dim(cv_coeff)[2]){
if(length(which(cv_coeff[,nn]==0))==0){nonzeroindx<-c(nonzeroindx,nn)}
  }
nonzero_coef<-data.frame(coef.name = names(trainfeat[nonzeroindx]),coef.value = colMeans(cv_coeff[,nonzeroindx]))

# sort coefficients in ascending order
selected_coef <- arrange(selected_coef,-selected_coef$coef.value)
nonzero_coef <- arrange(nonzero_coef,-nonzero_coef$coef.value)
nonzero_coef$coef.sd<-apply(cv_coeff[,nonzeroindx],2,sd)

# extract the top 10 and bottom 10 features
impsel_coef <- rbind(head(selected_coef,10),tail(selected_coef,10))
nonzerosel_coef <- rbind(head(nonzero_coef,10),tail(nonzero_coef,10))

ggplot(impsel_coef) + geom_bar(aes(x=reorder(impsel_coef$coef.name,impsel_coef$coef.value),y=impsel_coef$coef.value),stat="identity") +
  coord_flip() + ggtitle("Average Coefficents in the ELASTIC NET Model") +
  theme(axis.title=element_blank())

ggplot(nonzerosel_coef) + geom_bar(aes(x=reorder(nonzerosel_coef$coef.name,nonzerosel_coef$coef.value),y=nonzerosel_coef$coef.value),stat="identity") +
  coord_flip() + ggtitle("Average non-zero Coefficents in the ELASTIC NET Model") +
  theme(axis.title=element_blank())

# construct data frame for solution
#solution <- data.frame(Id=as.integer(rownames(X_test)),SalePrice=preds)
#write.csv(solution,"ridge_sol.csv",row.names=FALSE)

ggplot(nonzero_coef) + geom_bar(aes(x=reorder(nonzero_coef$coef.name,nonzero_coef$coef.value),y=nonzero_coef$coef.value),stat="identity",fill="#B9A1FF",colour="black") + geom_errorbar(aes(x=reorder(nonzero_coef$coef.name,nonzero_coef$coef.value),ymin=nonzero_coef$coef.value-nonzero_coef$coef.sd,ymax=nonzero_coef$coef.value+nonzero_coef$coef.sd),width=0.9)+
  +   coord_flip() + theme_classic()+
  +   theme(axis.title=element_blank())+theme(axis.text.x =element_blank())+theme(axis.text.y =element_blank())

#stuff
library(tidyverse)
library(MASS)
library(pROC)
library(ggplot2)
library(broom)
library(corrplot)
library(GGally)
library(ROCR)
#ML 
library(caret)
#rf
library(randomForest)


# Load 
#data <- read.csv('C:/Users/hosfa/Downloads/bank+marketing/bank-additional/bank-additional-full.csv', stringsAsFactors = T,sep=';')
data <- read.csv('C:/Users/hosfa/Downloads/bank+marketing/bank/bank-full.csv', stringsAsFactors = T,sep=';')
names(data)
#data <- subset(data,select=-c(pdays, nr.employed)) # data leakage
data <- subset(data,select=-pdays) # data leakage
prop.table(table(data$y))
summary(data)

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

#data$y <- as.factor(ifelse(data$y=='yes',1,0))
var_quali <- c('job','marital','education', 'default','housing','loan','contact','poutcome','month')
var_num <- c('age','balance','day','campaign','previous','duration')


# Univariée
par(mfrow=c(3,3))
for (variable in var_num) {
  plot(density(data[[variable]], na.rm = TRUE), main = paste("Densité de", variable), xlab = variable)
}


# Bivariée quali
GGally::ggbivariate(data, "y",
                    explanatory = var_quali) 

# bivariée quanti
for (var in var_num) {
  boxplot(data[[var]] ~ data$y, main = var, xlab = "Outcome", ylab = var)
}


#corrélations 
matrice_cor <- cor(data[, var_num])

# Création du graphique de la matrice de corrélation
corrplot(matrice_cor, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
#Echantillonnage 
set.seed(123) 
index <- createDataPartition(data$y, p=0.8, list=FALSE)
train <- data[index, ]
test <- data[-index, ]

cat("Proportion de la variable cible sur le jeu d'apprentissage")
prop.table(table(train$y))

cat("\nProportion de la variable cible sur le jeu de test")
prop.table(table(test$y))

#Regression logistique 
LR <- glm(y~., data= train, family = binomial)
summary(LR)


#proba
proba <- predict(LR, newdata = test, type = "response")
#pred
pred <- factor(ifelse(proba > 0.5, "positive","negative"))
print(table(pred))
#mc
mc <- table(test$y,pred)
mc
#tx erreur
err <- 1.0-sum(diag(mc))/sum(mc)
err*100

table(data$job)

# evaluation des variables 
res <- summary(LR)
res


# XGBoost
set.seed(123)
# splitting
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        savePredictions = "all",
                        summaryFunction = multiClassSummary,
                        returnResamp = "all") # return more metrics than binary classification

# parameter grid for XGBoost
parameterGrid <-  expand.grid(eta = c(0.001, 0.01, 0.1), # shrinkage (learning rate)
                              colsample_bytree = c(0.3, 0.5, 0.7), # subsample ration of columns
                              max_depth = c(1, 2, 3, 4), # max tree depth. model complexity
                              nrounds = c(30, 50, 70, 100), # boosting iterations
                              gamma = 0.1, # minimum loss reduction
                              subsample = 0.8, # ratio of the training instances
                              min_child_weight = c(2,3,4)) # minimum sum of instance weight

# parameter grid for random forest
# mtry = the number of features to use to build each tree
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 4))

model_xgb <- train(y~.,
                   data = train,
                   method = "xgbTree",
                   trControl = control,
                   tuneGrid = parameterGrid)
print(model_xgb)
#plot(model_xgb)

# Supposons que nous utilisons l'erreur de classification pour l'évaluation
results <- model_xgb$results
plot(results$nrounds, results$Accuracy, type = "b",
     xlab = "Number of Boosting Rounds",
     ylab = "Cross-Validated Accuracy",
     main = "Effect of Boosting Rounds on Model Accuracy")
model_xgb$bestTune

pred_xgb_raw <- predict(model_xgb,
                              newdata = test,
                              type = "raw")

pred_xgb_prob <- predict(model_xgb,
                               newdata = test,
                               type = "prob")

confusionMatrix(data = pred_xgb_raw,
                factor(test$y),
                positive = "yes")

varImpXGB <- varImp(model_xgb)
varImpXGB

# Maximiser la détection ( là où on maximise tpr et fpr)
set.seed(123)
probs <- pred_xgb_prob[, 2]
true_labels <- test$y  
roc_obj <- roc(true_labels, probs)
print(roc_obj$auc)
plot(roc_obj, main="Courbe ROC")
optimal_coords <- coords(roc_obj, "best", ret="threshold")
print(paste("Seuil optimal:", optimal_coords[,1]))
pred_class <- ifelse(probs > optimal_coords[,1], 1, 0)

true_labels <- factor(test$y, levels = c("no", "yes"))
pred_class <- factor(ifelse(pred_class == 1, "yes", "no"), levels = c("no", "yes"))

confusionMatrix(pred_class,true_labels)




# RandomForest

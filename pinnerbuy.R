library(Information)
library(caret)
library(mlbench)
library(randomForest)
library(rgl)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

folder = "/Users/xfang7/Documents/RCode/RAnalysis/"
user =  read.csv(paste(folder,"sessions.csv", sep=''))
df =  read.csv(paste(folder,"transactions.csv", sep=''))
df = merge(df,user,by.x="session_id",by.y="session_id")
df = df[!(df$test == TRUE & df$score == FALSE),]
rank = order(df$user_id)
df = df[rank,]
df$conversion = as.numeric(df$conversion)
table(df$conversion)
summary(df)
hist(df$num_impressions)
hist(df$num_search)
hist(df$avg_relevance)


sapply(subset(df,select = c("conversion","num_impressions","avg_relevance","num_search")), sd)
xtabs(~conversion + num_impressions, data = df)
xtabs(~conversion + num_search, data = df)

buy = df[df$conversion == TRUE,]
notBuy = df[df$conversion == FALSE,]
lu=length(unique(buy$user_id))
print(lu==nrow(buy))

summary(buy$avg_relevance)
summary(notBuy$avg_relevance)

summary(buy$num_impressions)
summary(notBuy$num_impressions)

summary(buy$num_search)
summary(notBuy$num_search)

df.train = df[df$train == TRUE,]
df.test = df[df$train == FALSE & df$score==TRUE,]
trainTrue = df.train[df.train$conversion == TRUE,]
for(i in nrow(df.test)){
  df.test[i,]$user_id %in% trainTrue$user_id
  print(paste(i,",traintrue,",df.test[i,]$conversion, df.test[i,]$user_id,sep=''))
  
}

correlationMatrix <- cor(df[,4:6])
print(correlationMatrix)


df.train$num_impressions = factor(df.train$num_impressions)
df.train$num_search = factor(df.train$num_search)

mylogit <- glm(conversion ~  avg_relevance + num_search, data = df.train, family = "binomial")
summary(mylogit)
confint(mylogit)
exp(coef(mylogit))
df$num_search = factor(df$num_search)
df.test$num_search = factor(df.test$num_search)
df$pred  = predict(mylogit, newdata = df, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df$conversion, df$pred )[1] 
df$pred = as.numeric(df$pred>= optCutOff)
mean(df$conversion == df$pred)
table(df$conversion,df$pred)
table(df$conversion, df$pred) %>% prop.table() %>% round(digits = 3)
anova(mylogit, test="Chisq")

df.test$pred  = predict(mylogit, newdata = df.test, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df.test$conversion, df.test$pred )[1] 
df.test$pred = as.numeric(df.test$pred>= optCutOff)
mean(df.test$conversion == df.test$pred)
table(df.test$conversion,df.test$pred)

df.train$pred  = predict(mylogit, newdata = df.train, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df.train$conversion, df.train$pred )[1] 
df.train$pred = as.numeric(df.train$pred>= optCutOff)
mean(df.train$conversion == df.train$pred)
table(df.train$conversion,df.train$pred)




predicted <- plogis(predict(mylogit, df))  # predicted scores
# or
df.test$num_search = factor(df.test$num_search)
df$num_search = factor(df$num_search)

predicted <- predict(mylogit, df, type="response")  # predicted scores
library(InformationValue)
optCutOff <- optimalCutoff(df$conversion, predicted)[1] 
pred = predicted >= round(optCutOff,2)
match = (pred == df$conversion)
sum(match)/length(df$conversion)


wald.test(b = coef(mylogit), Sigma = vcov(mylogit), Terms = 3:9)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(conversion ~ num_impressions +avg_relevance + num_search,  data=df.train, method="glm", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(df.train[,4:6], df.train[,2], sizes=c(4:6), rfeControl=control)
print(results)
plot(results, type=c("g", "o"))


set.seed(7)
mod_fit <- train(conversion ~ num_impressions +avg_relevance + num_search,  data=df.train, method="glm", family="binomial")

importance <- varImp(mod_fit,scale=FALSE)
print(importance)
plot(importance)


library(smbinning)
# segregate continuous and factor variables
continuous_vars <- c("num_impressions", "avg_relevance","num_search")
iv_df <- data.frame(VARS=c(continuous_vars), IV=numeric(3))  # init for IV results

# compute IV for categoricals
for(continuous_var in continuous_vars){
  smb <- smbinning(train, y="conversion", x=continuous_var)  # WOE table
  if(class(smb) != "character"){  # any error while calculating scores.
    iv_df[iv_df$VARS == continuous_var, "IV"] <- smb$iv
  }
}
iv_df <- iv_df[order(-iv_df$IV), ]  # sort
iv_df

logitMod <- glm(conversion~(num_impressions +avg_relevance), data=df.train, family=binomial(link="logit"))

predicted <- plogis(predict(logitMod, df))  # predicted scores
# or
predicted <- predict(logitMod, df.test, type="response")  # predicted scores
library(InformationValue)
optCutOff <- optimalCutoff(df$conversion, predicted)[1] 
pred = predicted >= round(optCutOff,2)
match = (pred == df$conversion)
sum(match)/length(df$conversion)
test = cbind(test,pred)

trainTrue = df.train[df.train$conversion == TRUE,]

for(i in nrow(test)){
  if(test[i,]$user_id )
    test[i,]$user_id %in% trainTrue$user_id
  test[i,]$pred = FALSE
  print(paste("train true,", i,sep=''))
  
}
match = pred == test$conversion
as.numeric(match)
accuracy = sum(match)/length(match)
print(accuracy)


# compute IV for continuous vars
# for(continuous_var in continuous_vars){
#   smb <- smbinning(trainingData, y="ABOVE50K", x=continuous_var)  # WOE table
#   if(class(smb) != "character"){  # any error while calculating scores.
#     iv_df[iv_df$VARS == continuous_var, "IV"] <- smb$iv
#   }
# }
# 
# iv_df <- iv_df[order(-iv_df$IV), ]  # sort
# iv_df


# 
# lr = glm(conversion~(num_impressions +avg_relevance+num_search),family=binomial(link="logit"),train)
# predictions <- predict(lr, test)
# 
# fit <- lda(conversion~(num_impressions +avg_relevance+num_search), data = train)
# predictions <- predict(fit, test)
# prediction.probabilities <- predictions$posterior[,2]
# predicted.classes <- predictions$class 
# observed.classes <- test$conversion
# accuracy <- mean(observed.classes == predicted.classes)
# 
# # remove redundant feature
# correlationMatrix <- cor(df[,4:6])
# print(correlationMatrix)
# highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print(highlyCorrelated)
# df$num_impressions
# 
# 
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
# model <- train(conversion~., data=train, method="lvq", preProcess="scale", trControl=control)
# importance <- varImp(model, scale=FALSE)
# print(importance)
# # plot importance
# plot(importance)
# 
# 
# lr = glm(train$conversion~.,family=binomial(link="logit"),train)
# 
# pred = fitted(lr) > 0.5
# as.numeric(pred)
# match = pred == train$conversion
# as.numeric(match)
# accuracy = sum(match)/length(match)
# print(accuracy)
# 
# 
# 
# woe =WOE(X= df.train$num_search,Y=df.train$conversion)
# options(scipen = 999, digits = 2)
# woetable= WOETable(X= df.train$num_search,Y=df.train$conversion)
# iv = IV(X= df.train$num_search,Y=df.train$conversion)
# q = quantile(df$num_impressions,probs = c())
# 
# 
# 
# 

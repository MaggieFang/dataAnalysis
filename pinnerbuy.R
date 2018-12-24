library(Information)
library(caret)
library(mlbench)
library(randomForest)
library(rgl)


folder = "/Users/xfang7/Documents/RCode/dataAnalysis/"
user =  read.csv(paste(folder,"sessions.csv", sep=''))
df =  read.csv(paste(folder,"transactions.csv", sep=''))
# merge two data file
df = merge(df,user,by.x="session_id",by.y="session_id")
# sort by userId
rank = order(df$user_id)
df = df[rank,]
df$conversion = as.numeric(df$conversion)

# get some information of data
table(df$conversion)
summary(df)

hist(df$num_impressions)
hist(df$num_search)
hist(df$avg_relevance)

sapply(subset(df,select = c("conversion","num_impressions","avg_relevance","num_search")), sd)
xtabs(~conversion + num_impressions, data = df)
xtabs(~conversion + num_search, data = df)

# divide into two file buy/not buy
buy = df[df$conversion == 1,]
notBuy = df[df$conversion == 0,]
lu=length(unique(buy$user_id))
print(lu==nrow(buy)) # buy的数量跟userId都是一样的7500，说明买的用户有买就是1次

# 观察train中true的user是否出现在test，结果只有一个
trainTrue = df.train[df.train$conversion == 1,]
for(i in nrow(df.test)){
  df.test[i,]$user_id %in% trainTrue$user_id
  print(paste(i,",traintrue,",df.test[i,]$conversion, df.test[i,]$user_id,sep=''))
}

# train data,test data,
df.train = df[df$train == TRUE,]
df.test = df[df$train == FALSE,]
df.fit = df[df$score == TRUE,]

#删除purchase后还推的数据
factor_train = factor(df.train$user_id)
level = levels(factor_train)
for(i in 1:length(level)){
   user_id = level[i]
   idx = which(df.train$user_id == user_id)
   data = df.train[idx,]
   data = data[order(data$session_dt),]
   for(j in 1:nrow(data)){
     if(data[j,]$conversion == 1 && j != nrow(data)){
       k = nrow(data)
       while( k >= j+1){
         print(paste("delete,",k,data[k,]$user_id,data[k,]$session_dt, sep=' '))
         df.train=df.train[-c(idx[k]),]  
         k= k-1
       }
       break
     }
   }
}

#相关性
correlationMatrix = cor(df[,4:6])
print(correlationMatrix)


# 把search认为是1-7的7个单独变量

df.train$num_search = factor(df.train$num_search)
mylogit <- glm(conversion ~  avg_relevance + num_search, data = df.train, family = "binomial")
summary(mylogit)
confint(mylogit)

# overall accuracy
df$num_search = factor(df$num_search)
df$pred  = predict(mylogit, newdata = df, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df$conversion, df$pred )[1] 
df$pred = as.numeric(df$pred>= optCutOff)
mean(df$conversion == df$pred)
table(df$conversion,df$pred)
anova(mylogit, test="Chisq")

# fitted accuracy
df.fit$num_search =factor(df.fit$num_search)
df.fit$pred  = predict(mylogit, newdata = df.fit, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df.fit$conversion, df.fit$pred )[1] 
df.fit$pred = as.numeric(df.fit$pred>= optCutOff)
mean(df.fit$conversion == df.fit$pred)
table(df.test$conversion,df.test$pred)

## train accuracy
df.train$pred  = predict(mylogit, newdata = df.train, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df.train$conversion, df.train$pred )[1] 
df.train$pred = as.numeric(df.train$pred>= optCutOff)
mean(df.train$conversion == df.train$pred)
table(df.train$conversion,df.train$pred)


### try cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(conversion ~ avg_relevance + num_search,  data=df.train, method="glm", preProcess="scale", trControl=control)
print(model)
# estimate variable importance
importance <- varImp(model, scale=FALSE)


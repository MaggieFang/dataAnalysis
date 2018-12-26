library(caret)
library(mlbench)

folder = "/Users/xfang7/Documents/RCode/dataAnalysis/"
user =  read.csv(paste(folder,"sessions.csv", sep=''))
df =  read.csv(paste(folder,"transactions.csv", sep=''))
# merge two data file
df = merge(df,user,by.x="session_id",by.y="session_id")
# sort by userId and then date
df = df[with(df,order(df$user_id,df$session_dt)),]

df$conversion = as.numeric(df$conversion)

# get some information of data
table(df$conversion)
hist(df$num_impressions,main = "Histogram of num_impression",xlab = "num_impression")
hist(df$num_search,main = "Histogram of num_search",xlab = "num_search")
hist(df$avg_relevance,main = "Histogram of avg_relevance",xlab = "avg_relevance")

x = xtabs(~conversion + num_search, data = df)
plot(x[1,], type="o", col="blue",axes=FALSE, ann=FALSE)
axis(1, at=1:8, lab=c("0", "1", "2", "3","4", "5","6","7"))
axis(2, las=1, at=1000*0:range(x)[2])
lines(x[2,], type="o", pch=22, lty=2, col="red")
title(main="conversion vs num_search", col.main="red", font.main=4)
title(xlab="num_search", col.lab=rgb(0,0.5,0))
title(ylab="total", col.lab=rgb(0,0.5,0))
legend("topright",c("conversion=0","conversion=1"), cex=0.8, col=c("blue","red"), pch=21:22, lty=1:2)

#相关性
correlationMatrix = cor(df[,4:6])
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)

# divide into two set  buy/not buy
buy = df[df$conversion == 1,]
buy_user_num=length(unique(buy$user_id))
not_buy_user_num= length(unique(df$user_id)) - buy_user_num
print(user_number_buy==nrow(buy)) # TRUE,user purchase 1 times, after that, even push again, he/she not buy any more

'%ni%' <- Negate('%in%')  # define 'not in' func
buy_detail = df[df$user_id %in% buy$user_id,]
not_buy_detail = df[df$user_id %ni% buy$user_id,]
buy_mean_sesion =  nrow(buy_detail)/buy_user_num
not_buy_mean_sesion =  nrow(not_buy_detail)/not_buy_user_num
#conclusion: user purchase make more search
buy_mean_search = sum(buy_detail$num_search)/buy_user_num
not_buy_mean_search = sum(not_buy_detail$num_search)/not_buy_user_num

buy_mean_relevance = sum(buy_detail$avg_relevance)/buy_user_num
not_buy_mean_relevance =sum(not_buy_detail$avg_relevance)/not_buy_user_num

hist(buy_detail$avg_relevance,main = "Histogram of avg_relevance for buy",xlab = "avg_relevance")
hist(not_buy_detail$avg_relevance,main = "Histogram of avg_relevance for not-buy",xlab = "avg_relevance")

# train data,test data,
df.train = df[df$train == TRUE,]
df.test = df[df$train == FALSE,]
df.fit = df[df$score == TRUE,]

mylogit = multinom(conversion~avg_relevance + num_search, data = df.train)

mylogit <- glm(conversion ~  avg_relevance + num_search , data = df.train, family = "binomial")
summary(mylogit)
confint(mylogit)
fit_row = nrow(df.fit)
for(i in 1:fit_row){
  data = df.fit[i,]
  idx = which(df$user_id == data$user_id)
  rows = df[idx,]
  testIdx = which(data$session_id == rows$session_id)
  rows = rows[1:testIdx,]
  df.fit[i,]$num_search = sum(rows$num_search)
}
df.fit$pred  = predict(mylogit, newdata = df.fit, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df.fit$conversion, df.fit$pred )[1] 
df.fit$pred = as.numeric(df.fit$pred>= optCutOff)
mean(df.fit$conversion == df.fit$pred)
table(df.fit$conversion,df.fit$pred)
df.fit$acc = (df.fit$conversion == df.fit$pred)

df$pred  = predict(mylogit, newdata = df, type = "response")
library(InformationValue)
optCutOff <- optimalCutoff(df$conversion, df$pred )[1] 
df$pred = as.numeric(df$pred>= optCutOff)
mean(df$conversion == df$pred)
table(df$conversion,df$pred)


#######################以下是之前尝试#######################

# 观察train中true的user是否出现在test，结果只有一个
trainTrue = df.train[df.train$conversion == 1,]
for(i in nrow(df.test)){
  df.test[i,]$user_id %in% trainTrue$user_id
  print(paste(i,",traintrue,",df.test[i,]$conversion, df.test[i,]$user_id,sep=''))
}


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


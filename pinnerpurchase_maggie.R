library(caret)
library(ROCR)
library(randomForest)

setwd()
user =  read.csv("sessions.csv")
df =  read.csv("transactions.csv")
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

#Correlation
correlationMatrix = cor(df[,4:6])
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)

fit_rf = randomForest(conversion~avg_relevance + num_search , data=df)
importance(fit_rf)
varImpPlot(fit_rf)

# divide into two set buy/not buy
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
df.fit = df[df$score == TRUE,]

# define training control
train_control = trainControl(method = "cv", number = 10)
mylogit = train(conversion~avg_relevance + num_search , data = df.train,trControl = train_control,method = "glm",family=binomial())
# mylogit = glm(conversion~avg_relevance + num_search , data = df, family = "binomial",subset = df.train)
summary(mylogit)

confint(mylogit)

fit_row = nrow(df.fit)
for(i in 1:fit_row){
  data = df.fit[i,]
  idx = which(df$user_id == data$user_id)
  rows = df[idx,]
  testIdx = which(data$session_id == rows$session_id) # ignore the sessions after test, since we only use history not future to predict.
  rows = rows[1:testIdx,]
  df.fit[i,]$num_search = sum(rows$num_search)
}

df.fit$pred  = predict(mylogit, newdata = df.fit, type = "response")
hist(df.fit$pred)
pred= prediction(df.fit$pred,df.fit$conversion)
eval = performance(pred,"acc")
plot(eval)
#identify best values
max=which.max(slot(eval,"y.values")[[1]])
acc = slot(eval,"y.values")[[1]][max]
cut = slot(eval,"x.values")[[1]][max]
df.fit$pred = as.numeric(df.fit$pred>= cut)
abline(h = acc, v= cut)
mean(df.fit$pred == df.fit$conversion)
caret::confusionMatrix(factor(df.fit$conversion),factor(df.fit$pre),positive = "1")
table(df.fit$conversion,df.fit$pred)

p = predict(mylogit, newdata=df.fit, type = "response")
pr = prediction(p, df.fit$conversion)
roc = performance(pr,"tpr","fpr")
plot(roc,colorize = T,main= "ROC Curve")
abline(a=0,b=1)
auc = performance(pr,"auc")
auc = unlist(slot(auc,"y.values"))
auc = round(auc,4)
legend(.6,.3,auc,title = "AUC")

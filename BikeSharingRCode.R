##
## Analyzing Bike Sharing Dataset
##

library(caret)
library(vroom) 
library(DataExplorer)
library(plyr)
library(h2o)
library(tidyverse)

# Read in data
bike.test <- vroom("Desktop/Kaggle/KaggleBikeShare/bike-sharing-demand/test.csv")
bike.train <- vroom("Desktop/Kaggle/KaggleBikeShare/bike-sharing-demand/train.csv")
bike <- bind_rows(train=bike.train, test=bike.test, .id="id")

# Drop casual and registered columns
bike <- bike %>% select(-casual, -registered)

# Feature Engineering

# split datetime into separate columns for year, month, day, and hour
library(lubridate)
bike$year <- year(bike$datetime) %>% as.factor()
bike$month <- month(bike$datetime) %>% as.factor()
bike$day <- day(bike$datetime) %>% as.factor()
bike$hour <- hour(bike$datetime) %>% as.factor()

# add weekday column
bike$weekday = wday(bike$datetime, label=TRUE)

# factorize and convert season from numerical to labeled
bike$season <- as.factor(bike$season)
#bike$season = revalue(bike$season, c("1"="Spring", "2"="Summer", "3"="Fall", "4"="Winter"))

# Exploratory Plots
qplot(1:nrow(bike), bike$count, geom="point")
ggplot(data=bike, aes(x=datetime, y=count, color=as.factor(month(datetime)))) + geom_point()

plot_missing(bike)

# atemp and temp confound so we will only select one to use while training the data
plot_correlation(bike, type="continuous", cor_args=list(use="pairwise.complete.obs"))

ggplot(data=bike, aes(x=season, y=count)) + 
  geom_boxplot()

# Target encoding

# go to category, ask what is the average response variable in that category
# go to spring, calculate the average count in spring, put that in for the category 
# go to summer, calculate the average count in summer, put that in for the category
# linear model with categorical variable provides only averages of each category
# this gives the average count within each season
bike$season <- lm(count~season, data=bike) %>% predict(., newdata=bike %>% select(-count))

table(bike$season)

## Fit models

tr = trainControl(method="repeatedcv", number=3, repeats=3, search="grid")

### XGBoost Tree Model ###
xgbTreeGrid <- expand.grid(nrounds = c(1, 10),
                       max_depth = c(1, 4),
                       eta = c(.1, .4),
                       gamma = 0,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = c(.8, 1)) 

# taking the log of count so the model will only predict positive values
# using columns only included in the original dataset
bike.model <- train(form=log(count)~season+holiday+workingday+weather+temp+humidity,
                             data = bike %>% filter(id=="train"),
                             method = "xgbTree",
                             metric = "RMSE",
                             tuneGrid = xgbTreeGrid,
                             gamma = 0.5,
                             trControl=tr)

plot(bike.model)
# exponentiate the log results to transform them back to correct values
preds <- exp(predict(bike.model, newdata=bike %>% filter(id=="test")))
submission <- data.frame(datetime=bike %>% filter(id=="test") %>% pull(datetime),
                         count=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/Kaggle/KaggleBikeShare/xbgTreeSubmit")

### XGBoost Linear Model ###
xgbLinearGrid <- expand.grid(nrounds = c(1, 10),
                       lambda = c(.01, .1),
                       alpha = c(.01, .1),
                       eta = .3)

# taking the log of count so the model will only predict positive values
# using columns only included in the original dataset
bike.model <- train(form=log(count)~season+holiday+workingday+weather+temp+humidity,
                             data = bike %>% filter(id=="train"),
                             method = "xgbLinear",
                             metric = "RMSE",
                             tuneGrid = xgbLinearGrid,
                             gamma = 0.5,
                             trControl=tr)
plot(bike.model)
# exponentiate the log results to transform them back to correct values
preds <- exp(predict(bike.model, newdata=bike %>% filter(id=="test")))
submission <- data.frame(datetime=bike %>% filter(id=="test") %>% pull(datetime),
                         count=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/Kaggle/KaggleBikeShare/xgbLinearSubmit")

### GBM_h2o
h2o.init()

# fit gbm_h2o model using default parameters
# additional features used are year, month, day, hour, and weekday
bike.model <- train(form=log(count)~season+holiday+workingday+weather+temp+humidity+year+month+day+hour+weekday,
                             data = bike %>% filter(id=="train"),
                             method = "gbm_h2o",
                             metric = "RMSE",
                             trControl=tr)
preds <- exp(predict(bike.model, newdata=test))
submission <- data.frame(datetime=bike %>% filter(id=="test") %>% pull(datetime),
                         count=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/Kaggle/KaggleBikeShare/gbm_h2oSubmit")

#gbmGrid <- expand.grid(ntrees=150, 
#                       max_depth=6, 
#                       min_rows=100, 
#                       learn_rate=0.01, 
#                       col_sample_rate=1)

# use the following tuning parameters on gbm_h2o model
# tried to find a balance between ntrees, depth, min_rows, and learning rate
# that will provide a model which does not overfit but is general enough
# to maintain a good accuracy
gbmGrid <- expand.grid(ntrees=70, 
                       max_depth=10, 
                       min_rows=15, 
                       learn_rate=0.1, 
                       col_sample_rate=1)

bike.model <- train(form=log(count)~season+holiday+workingday+weather+temp+humidity+year+month+day+hour+weekday,
                             data = bike %>% filter(id=="train"),
                             method = "gbm_h2o",
                             metric = "RMSE",
                             tuneGrid = gbmGrid,
                             trControl=tr)

# these tuning parameters on the gbm_h2o model resulted in a score of 0.44437 on the Kaggle Bike Sharing Competition

preds <- exp(predict(bike.model, newdata=test))
submission <- data.frame(datetime=bike %>% filter(id=="test") %>% pull(datetime),
                         count=preds)
write.csv(x=submission, row.names=FALSE, file="~/Desktop/Kaggle/KaggleBikeShare/gbm_h2oSubmit")


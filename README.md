# Kaggle-Bike-Share
Analysis of Kaggle Bike Sharing prediction competition. https://www.kaggle.com/c/bike-sharing-demand/overview

### Purpose
This is my first time playing around with a Kaggle competition and I've decided to try out the Bike Sharing Demand contest which is fairly straightforward with a nice and clean data set. My goal is to familiarize myself more with fitting machine learning models, specifically using R.

### Files
The bike-sharing-demand folder contains train and test files provided from the competition as well as a sample submission file. 

Each submission file contains the count predictions from different models used in my analysis. 

The R script contains code with an overview of my analysis including data exploration, feature engineering, and model selection and tuning.
	
### Methods
The feature engineering I did on this data set mainly came from splitting the datetime object into separate variables using the lubridate library. I created separate variables for year, month, day, hour, and weekday. These were the only additional features I used in my analysis.

### Models
The models I used during my prediction were the eXtreme Gradient Boosting linear/tree from the xgboost library as well as a Gradient Boosting Machine from the h2o library. The highest accuracy I achieved was with h2o’s gradient boosting machine. After including variables created through feature engineering and some parameter tuning I was able to achieve a score of 0.444 which is in the top 25% percentile of scores submitted during the competition. 

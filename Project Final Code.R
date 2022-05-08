#Title: "BDA Project"
#Group Name: The Debuggers
#TIME TAKEN: 4 Seconds to run the selected model, Linear regression with predictions and approximately 23.7 seconds to run the entire script with the three tried models

# load packages  (please deomment incase these packages are not loaded in your system, we commented as everytime we ran the entire script, it would take a lot of time to run)
#install.packages('corrplot')
#install.packages('ggplot2')
#install.packages('caret', dependencies=TRUE) #randomforest
#install.packages('randomForest')
#install.packages('glmnet') #lasso

#Load the libraries
library(readr)
library(ggplot2)
library(repr)
library(corrplot)
library(randomForest)
library(glmnet)
#library(rpart)
#library(leaps)
library(caret)
options(scipen=999)

# read data from a csv file 
df = read.csv('historic_property_data.csv')
head(df)
#See Column names of the dataframe
names(df)
#see dimensions of the CSV File
dim(df)

#Cleaning the Data by handling NA values by Data Imputation

sum(is.na(df))
df$char_fbath[is.na(df$char_fbath)] <- 1
#replacing char_fbath null values by 1 as suggested by the codebook of the data.
names<-(colnames(df)[colSums(is.na(df)) > 0])  #Identifying columns having more than NA value
print(((names)))

#replacing numerical variables
all_numVar <- names(Filter(is.numeric, df))
for(name in all_numVar) { 
  temp <- df[, c(name)]  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  df[, c(name)] <- temp 
} 
sum(is.na(df))

#Replacing NA in character variables
contvarnames <- names(Filter(is.character, df))
contvarnames
for(name in contvarnames) { 
  temp <- df[, c(name)]  
  temp[is.na(temp)]<- mode(temp) 
  df[, c(name)] <- temp 
} 

#replacing NA values in logical variables
contvarnames <- names(Filter(is.logical, df))
contvarnames
for(name in contvarnames) { 
  temp <- df[, c(name)]  
  temp[is.na(temp)]<- mode(temp) 
  df[, c(name)] <- temp 
} 

sum(is.na(df)) # zero NA Values shown

#Data Visualization
options(repr.plot.width=10, repr.plot.height=8)
plot(x=df$sale_price, y=df$meta_certified_est_bldg, pch=18, cex=2, col="orange",cex.main=1.5, cex.lab = 1.5,cex.axis=1, xlab="Meta Certified East Buiding", ylab="Sale Price", main="Scatter plot showing Sale Price vs Meta_certified_est_bldg") 

options(repr.plot.width=10, repr.plot.height=10)
bp1 <- ggplot(df, aes(x = char_air, fill =char_air )) + 
  geom_bar(width = 0.5)+ 
  scale_fill_hue(c = 80)+
  ggtitle("Houses by the Air Conditioning")+
  ylab("Frequency") +
  xlab("Type of Air Conditioned Rooms") +
  theme(plot.title = element_text(hjust = 0.5, size=20),legend.position="right", legend.background = element_rect(fill="Blue", size=0.5, linetype="solid", colour ="black"))+
  geom_text(stat='count',aes(label=..count..),vjust=-0.25)+ 
  theme(text = element_text(size = 20)) 
bp1

#Applying Correlation to identify highly correlated variables so that they can be removed
all_numVar <- names(Filter(is.numeric, df))
cor(df[all_numVar]) 

#Correlation Plot
options(repr.plot.width=50, repr.plot.height=25)
corrplot(cor(df[all_numVar]), method = "circle",col = topo.colors(100) )

#df2 = cor(df[all_numVar])
#hc = findCorrelation(df2, cutoff=0.8) # put any value as a "cutoff" 
#hc = sort(hc)
#reduced_Data = df[,-c(hc)]
#head(reduced_Data)


#Removing the not important Variables acc to correlation done above
drop <- c("geo_property_city","geo_property_zip","geo_school_elem_district","geo_school_hs_district","census_tract","MAILING_ADDRESS","MAILING_STATE","MAILING_CITY","MAILING_ZIP","DOC_NO","ind_arms_length","ind_garage","ind_large_home")
df = df[,!(names(df) %in% drop)]


#Dividing the Dataset into Training and Testing
set.seed(1)

# the total number of rows  
dim(df)[1]

# row numbers of the training set 
dim(df)[1]*0.6

# training set 
train.index <- sample(c(1:dim(df)[1]), dim(df)[1]*0.6)  #selecting the rows to choose
train.df <- df[train.index,]

# test set
#test.df <- df[-train.index,]
#dim(test.df)

test.df <- df[-train.index,]
test.index <- sample(c(1:dim(test.df)[1]), dim(test.df)[1]*1)  #selecting the rows to choose
dim(test.df)
#head(test.df)


#METHOD CHOSEN : Linear Regression
#Training the model
sqft_model <- lm(formula=sale_price~.,data=train.df)

#Fitting the model and making predictions on the Test Data
real_test <- test.df$sale_price
predict_test <- predict(sqft_model,test.df)

#Calculating the MSE
mean((test.df$sale_price-predict_test)^2)

#LOADING THE PREDICTOR FILE
pred_df = read.csv('predict_property_data.csv')
head(pred_df)

#dropping the same columns we dropped for the Training Data
drop <- c("geo_property_city","geo_property_zip","geo_school_elem_district","geo_school_hs_district","census_tract","MAILING_ADDRESS","MAILING_STATE","MAILING_CITY","MAILING_ZIP","DOC_NO","ind_arms_length","ind_garage","ind_large_home")
pred_df = pred_df[,!(names(pred_df) %in% drop)]

#Handling NA Values of the Prediction Data
sum(is.na(pred_df))
pred_df$char_fbath[is.na(pred_df$char_fbath)] <- 1
#replacing char_fbath null values by 1 as suggested by the codebook of the data.
names<-(colnames(pred_df)[colSums(is.na(pred_df)) > 0])  #Identifying columns having more than NA value
print(((names)))

#replacing numerical variables
all_numVar <- names(Filter(is.numeric, pred_df))
for(name in all_numVar) { 
  temp <- pred_df[, c(name)]  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  pred_df[, c(name)] <- temp 
} 
sum(is.na(pred_df))

#Replacing NA in character variables
contvarnames <- names(Filter(is.character, pred_df))
contvarnames
for(name in contvarnames) { 
  temp <- pred_df[, c(name)]  
  temp[is.na(temp)]<- mode(temp) 
  pred_df[, c(name)] <- temp 
} 

#replacing NA values in logical variables
contvarnames <- names(Filter(is.logical, pred_df))
contvarnames
for(name in contvarnames) { 
  temp <- pred_df[, c(name)]  
  temp[is.na(temp)]<- mode(temp) 
  pred_df[, c(name)] <- temp 
} 

sum(is.na(pred_df))

#Predictions
predict_test <- predict(sqft_model,pred_df)
head(predict_test)
summary(predict_test)

#Saving predictions into a CSV
write.csv(predict_test, file = "assessed_value.csv")

##--------------------LASSO REGRESSION --------------------

# convert a data frame of predictors to a matrix and create dummy variables for character variables 
x <- model.matrix(sale_price~.,df)[,-1]
# first six rows of x
head(x)
# outcome 
y <- df$sale_price
is.vector(y) #to check if y is a vector
y.test <- y[test.index] # outcome in the test set 

# fit a lasso regression model 
library(glmnet)
fit<- glmnet(x[train.index,],y[train.index],alpha=1)

# sequence of lambda values 
fit$lambda
#lambda.small <-fit$lambda[1]
#lambda.medium <-fit$lambda[20]
#lambda.large <-fit$lambda[50]

# dimension of lasso regression coefficients 
dim(coef(fit))

# Return a medium lambda value 
lambda.medium <-fit$lambda[20]
lambda.medium
# lasso regression coefficients  
coef.lambda.medium <- predict(fit,s=lambda.medium,type="coefficients")[1:20,]
coef.lambda.medium
# non-zero coefficient estimates  
coef.lambda.medium[coef.lambda.medium!=0]
# make predictions for records the test set 
pred.lambda.medium <- predict(fit,s=lambda.medium,newx=x[test.index,])
head(pred.lambda.medium)
# MSE in the test set 
mean((y.test-pred.lambda.medium)^2) 

##----------------------RANDOM FOREST---------------------

contvarnames <- names(Filter(is.numeric, df))
contvarnames

names<-(colnames(df)[colSums(is.na(df)) > 0]) 

for(name in contvarnames) { 
  print(name)
  temp <- df[, c(name)]  
  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  
  df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
} 


all_numVar <- names(Filter(is.numeric, df))
all_numVar
names<-(colnames(df)[colSums(is.na(df)) > 0])
for(name in contvarnames) { 
  print(name)
  temp <- df[, c(name)]  
  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  
  df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
} 

contvarnames <- names(Filter(is.character, df))
contvarnames
for(name in contvarnames) { 
  print(name)
  temp <- df[, c(name)]  
  
  temp[is.na(temp)]<- mode(temp) 
  
  df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
} 

contvarnames <- names(Filter(is.logical, df))
contvarnames
for(name in contvarnames) { 
  print(name)
  temp <- df[, c(name)]  
  
  temp[is.na(temp)]<- mode(temp) 
  
  df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
}

sum(is.na(df))

set.seed(1)
# row numbers of the training set
train.index <- sample(c(1:dim(df)[1]), dim(df)[1]*0.6)  
head(train.index)
# training set 
train.df <- df[train.index, ]
head(train.df)
# test set 
test.df <- df[-train.index, ]
head(test.df)


#random Forest implementation
set.seed(1)
rf <- randomForest(sale_price ~ meta_class + meta_town_code + meta_nbhd + meta_certified_est_bldg + meta_certified_est_land + meta_cdu + meta_deed_type + char_hd_sf + char_age + char_apts + char_ext_wall + char_roof_cnst + char_rooms + char_beds + char_bsmt + char_bsmt_fin + char_heat + char_oheat + char_air + char_frpl + char_attic_type + char_fbath + char_hbath + char_tp_plan + char_tp_dsgn + char_cnst_qlty + char_site + char_gar1_size + char_gar1_cnst + char_gar1_att + char_gar1_area + char_ot_impr + char_bldg_sf + char_repair_cnd + char_use + char_type_resd + char_attic_fnsh + char_renovation + char_porch + geo_tract_pop + geo_white_perc + geo_black_perc + geo_asian_perc + geo_his_perc + geo_other_perc + geo_fips + geo_ohare_noise + geo_municipality + geo_floodplain + geo_fs_flood_factor + geo_fs_flood_risk_direction + geo_withinmr100 + geo_withinmr101300 + econ_tax_rate + econ_midincome , data=train.df, mtry = 4, ntree=10)

#Prediction on Test Data
y_pred <- predict(rf, newdata = test.df)
y_pred

#Calculating MSE
mean((test.df$sale_price -y_pred)^2)

#Cleaning the Data
contvarnames <- names(Filter(is.numeric, predict_df))
contvarnames

names<-(colnames(predict_df)[colSums(is.na(predict_df)) > 0]) 

for(name in contvarnames) { 
  print(name)
  temp <- predict_df[, c(name)]  
  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  
  predict_df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
}


all_numVar <- names(Filter(is.numeric, predict_df))
all_numVar
names<-(colnames(predict_df)[colSums(is.na(predict_df)) > 0])
for(name in contvarnames) { 
  print(name)
  temp <- predict_df[, c(name)]  
  
  temp[is.na(temp)]<- mean(temp,na.rm=TRUE) 
  
  predict_df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
}


contvarnames <- names(Filter(is.character, predict_df))
contvarnames
for(name in contvarnames) { 
  print(name)
  temp <- predict_df[, c(name)]  
  
  temp[is.na(temp)]<- mode(temp) 
  
  predict_df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
}

contvarnames <- names(Filter(is.logical, predict_df))
contvarnames
for(name in contvarnames) { 
  print(name)
  temp <- predict_df[, c(name)]  
  
  temp[is.na(temp)]<- mode(temp) 
  
  predict_df[, c(name)] <- temp 
  print(temp[is.na(temp)] )
}

sum(is.na(predict_df))


#PREDICTING VALUES
y_pred <- predict(rf, newdata = predict_df)
y_pred

# Webstore Sales Forecasting by Jim Frank

### _Facebook Prophet & LSTM Model Forecasting_  

_(LSTM : long short term memory)_  
_Webstore data from  [Retailrocket recommender system dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset) ._  
_This project builds upon my past project with this data  [Web Customer Fingerprints](https://github.com/truejimfrank/WebCustomerFingerprints) ._

---

## Table of Contents
1. [Objective](#objective)
2. [The Data](#the-data)
3. [Methodology](#methodology)
4. [Facebook Prophet Model](#facebook-prophet-model)
5. [LSTM Model](#lstm-model)
6. [Product Category Forecasts](#product-category-forecasts)
7. [Forecast Wrap Up](#forecast-wrap-up)
8. [References](#references)

---

## Objective

<b>WHY FORECASTING:  </b> 
Forecasting can provide answers to common questions in business.  
Additionally, it is flexible, practical, and can be applied to many different objectives.
A common question such as:

<b>QUESTION:  </b> 
Next month, what product category will have the best sales growth?  

<b>GOALS:  </b> 
1. BUILD forecasting models
2. EVALUATE forecasting models
3. FORECAST with the distinction being product categories

## The Data

### Context

The data has been collected from a real-world ecommerce website. It is raw data, i.e. without any content transformations, however, all values are anonymous.

### Content

Behaviour data consists of 3 event types. Those being **views, add to carts, and transactions.** These product interactions were collected over a period of 4.5 months in 2015.

Here's an example of the raw data:

| timestamp | visitorid | event | itemid | transactionid |
|--|--|--|--|--|
| 2015-06-01 23:02:12 | 257597 | view | 355908 | NaN |
| 2015-06-01 23:50:14 | 992329 | view | 248676 | NaN |

### EDA (Exploratory Data Analysis)

![eda events bar](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/events_hbar.png)

**118** product views for every **1** product purchase  

**236** unnamed product categories  

![user product hist](https://github.com/truejimfrank/WebCustomerFingerprints/blob/master/img/product_hist.png)

<sub><b>Figure: </b> Count of visitors binned on # products interacted with </sub>

It is important to note that daily purchases is simply a count of products sold each day. It is not connected with product pricing or revenue.  

![data timeseries](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/data_timeseries.png)

<sub><b>Figure: </b> Daily purchases for the full dataset </sub>

## Methodology

I would like the forecast outputs of the tools I work with to actually be accurate and insightful. To make sure this happens, I have followed this methodology. All steps are important for reaching the accuracy I desire.

1. BUILD  
2. EVALUATE  
3. FORECAST  

## Facebook Prophet Model

Here’s the first model I built.     

CHANGE LINK
![FB Prophet full model](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/data_timeseries.png)

<sub><b>Figure: </b> FB Prophet built with the full dataset </sub>

The forecasting tool I used is called [Facebook Prophet](http://facebook.github.io/prophet/). A team at Facebook introduced this tool about 2.5 years ago. I read their technical paper and really liked their approach to forecasting. I liked it so much that I decided to use it for this project. Simply put, the way it works goes like this…  Trend + Seasonal Components = Forecast  
Based on the amount of variance in the data, confidence intervals are generated.

![weekly](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/weekly.png)

<sub><b>Figure: </b> Weekly component of the model. Negative values because this component is added to the trend </sub>

## LSTM Model

To have another model for comparison, I made an LSTM neural net model with TensorFlow / Keras.

![LSTM Initial Forecast](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/RNN_initial_result.png)

![LSTM Best Forecast](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/weekly.png)

## Product Category Forecasts

I compared the FB Prophet model and the LSTM model based on there results and the amount of time I spent getting the models to make forecasts.  
FB Prophet is fast and works well with my data, so I used it going forward.

![Cherries Category](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/weekly.png)

![Holiday Category](https://github.com/truejimfrank/StreamlinedProductPurchaseForecasting/blob/master/img/weekly.png)

## Forecast Wrap Up

Today we've only had time for one cluster model paired with one regression classifier. Thankfully, there are other paths of investigation to choose from in the data science toolbelt.


## References


# Sentiment-Analysis-of-Text-Data-Tweets-
This project addresses the problem of sentiment analysis on Twitter. The goal of this project was to predict sentiment for the given Twitter post using Python. Sentiment analysis can predict many different emotions attached to the text, but in this report, only 3 major were considered: positive, negative and neutral. The training dataset was small (just over 5900 examples) and the data within it was highly skewed, which greatly impacted on the difficulty of building a good classifier. After creating a lot of custom features, utilizing bag-of-words representations and applying the Extreme Gradient Boosting algorithm, the classification accuracy at the level of 58% was achieved. Analysing the public sentiment as firms trying to find out the response of their products in the market, predicting political elections and predicting socioeconomic phenomena like the stock exchange.

## Dataset Information

We use and compare various different methods for sentiment analysis on tweets . The training dataset is expected to be a csv file of type `tweet_id,sentiment,tweet` where the `tweet_id` is a unique integer identifying the tweet, `sentiment` is either `1` (positive) or `0` (negative), and `tweet` is the tweet enclosed in `""`. Similarly, the test dataset is a csv file of type `tweet_id,tweet`. Please note that csv headers are not expected and should be removed from the training and test datasets.
The input data consisted of two CSV files:
* train.csv (5971 tweets)
* test.csv (4000 tweets)

## Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.
* `python 3x`
* `pandas`
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`
* `regex`
* `matplotlib`
* `seaborn`

The library requirements specific to some methods are:
* `xgboost` for XGBoost.

**Note**: It is recommended to use Anaconda distribution of Python. It already have lot of prebuild libraries installed.

## Information about files

* `data/text.csv`: Training Dataset.
* `data/train.csv`: Testing Dataset.
* `data/emoticons.txt`: Emoticons Dataset.
* `Sentiment Analysis of Text Data.ipynb`: Heart of the beast, contains all the functions, models and results from the project.
* `report.pdf`: report of the project.
* `proposal.pdf`: proposal of the project.
* `plots`: consists of plots produced in project.
* `Sentiment Analysis of Text Data.html`: Jupyter notebook in html format.
* `result.csv`: result of the test data.

 ## How to Run the project

* For the sake of simplicity I have added everything in a single Python Notebook file. Follow the Notebook, each cell in the notebook is 
well commented which will help understand the project steps.


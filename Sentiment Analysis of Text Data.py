#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# 

# ## Sentiment Analysis of Text Data
# 

# ## Import  Libraries

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
import nltk
import pandas as pd
#from emoticons import EmoticonDetector
import re as regex
import numpy as np
#import plotly
#from plotly import graph_objs
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#plotly.offline.init_notebook_mode()

import seaborn as sns
import plotly
import cufflinks as cf
import re
nltk.download('punkt')


# ## Import the Data

# In[2]:


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data.rename(columns={'Category': 'emotion'}, inplace=True)
test_data.rename(columns={'Category': 'Tweet'}, inplace=True)

train_data = train_data[train_data['emotion'] != 'Tweet']


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


test_data.head()


# In[6]:


test_data.info()


# # Data visualization

# ## Distribution of target class

# In[7]:


sns.countplot(x='emotion',data=train_data)


# In[ ]:





# ## Data Cleaning

# In[8]:


# remove the tweets which contains Not available

train_data = train_data[train_data['Tweet'] != "Not Available"]

### Function to clean tweets
Remove URLs
Remove usernames (mentions)
Remove tweets with Not Available text
Remove special characters
Remove numbers

# In[9]:


def clean_tweets(tweet):
    
    # remove URL
    tweet = re.sub(r"http\S+", "", tweet)
    
    # Remove usernames
    tweet = re.sub(r"@[^\s]+[\s]?",'',tweet)
    
    # remove special characters 
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)
    
    # remove Numbers
    tweet = re.sub('[0-9]', '', tweet)
    
    return tweet


# In[10]:


# Apply function to Tweet column

train_data['Tweet'] = train_data['Tweet'].apply(clean_tweets)


# In[11]:


'''
text = 'text4 http://url.com/bla2/blah2'
re.sub(r"http\S+", "", text)
text = '@ajay dkfhskf dfs'
re.sub(r"@[^\s]+[\s]?",'',text)
re.sub('[^ a-zA-Z0-9]', '', text)
'''


# In[12]:


train_data['Tweet'].head()


# In[ ]:





# ## Tokenization & stemming

# In[13]:


# Function which directly tokenize the tweet data
from nltk.tokenize import TweetTokenizer

tt = TweetTokenizer()
train_data['Tweet'].apply(tt.tokenize)


# In[14]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()


# In[15]:


def tokenize(text):
    return word_tokenize(text)

def stemming(words):
    stem_words = []
    for w in words:
        w = ps.stem(w)
        stem_words.append(w)
    
    return stem_words


# In[16]:


# apply tokenize function
train_data['text'] = train_data['Tweet'].apply(tokenize)


# In[17]:


# apply steming function
train_data['tokenized'] = train_data['text'].apply(stemming)


# In[18]:


train_data.head()


# In[ ]:





# ## Wordlist

# In[19]:


words = Counter()
for idx in train_data.index:
    words.update(train_data.loc[idx, "text"])

words.most_common(5)


# In[20]:


nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words("english")


# In[21]:


whitelist = ["n't", "not"]
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]
words.most_common(5)


# In[22]:


def word_list(processed_data):
    #print(processed_data)
    min_occurrences=3 
    max_occurences=500 
    stopwords=nltk.corpus.stopwords.words("english")
    whitelist = ["n't","not"]
    wordlist = []
    
    whitelist = whitelist if whitelist is None else whitelist
    #print(whitelist)
    '''
    import os
    if os.path.isfile("wordlist.csv"):
        word_df = pd.read_csv("wordlist.csv")
        word_df = word_df[word_df["occurrences"] > min_occurrences]
        wordlist = list(word_df.loc[:, "word"])
        #return
    '''
    words = Counter()
    for idx in processed_data.index:
        words.update(processed_data.loc[idx, "text"])

    for idx, stop_word in enumerate(stopwords):
        if stop_word not in whitelist:
            del words[stop_word]
    #print(words)

    word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                                 "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
                           columns=["word", "occurrences"])
    #print(word_df)
    word_df.to_csv("wordlist.csv", index_label="idx")
    wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]
    #print(wordlist)


# In[23]:


word_list(train_data)


# In[24]:


words = pd.read_csv("wordlist.csv")


# ## Bag of Words

# In[25]:


import os


# In[26]:


wordlist= []
if os.path.isfile("wordlist.csv"):
    word_df = pd.read_csv("wordlist.csv")
    word_df = word_df[word_df["occurrences"] > 3]
    wordlist = list(word_df.loc[:, "word"])

label_column = ["label"]
columns = label_column + list(map(lambda w: w + "_bow",wordlist))
labels = []
rows = []
for idx in train_data.index:
    current_row = []
    
    # add label
    current_label = train_data.loc[idx, "emotion"]
    labels.append(current_label)
    current_row.append(current_label)

    # add bag-of-words
    tokens = set(train_data.loc[idx, "text"])
    for _, word in enumerate(wordlist):
        current_row.append(1 if word in tokens else 0)

    rows.append(current_row)

data_model = pd.DataFrame(rows, columns=columns)
data_labels = pd.Series(labels)


bow = data_model


# ## Classification

# In[27]:


import random
seed = 777
random.seed(seed)


# In[28]:


def test_classifier(X_train, y_train, X_test, y_test, classifier):
    log("")
    log("---------------------------------------------------------")
    log("Testing " + str(type(classifier).__name__))
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    log("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    log("Predicting time {0}s".format(time() - now))

    # Calculate Accuracy, Precision, recall
    
    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    
    log("=================== Results ===================")
    log("            Negative     Neutral     Positive")
    log("F1       " + str(f1))
    log("Precision" + str(precision))
    log("Recall   " + str(recall))
    log("Accuracy " + str(accuracy))
    log("===============================================")

    return precision, recall, accuracy, f1

def log(x):
    #can be used to write to log file
    print(x)


# ## Experiment 1: BOW + Naive Bayes

# In[29]:


from sklearn.naive_bayes import BernoulliNB
X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow['label'], test_size=0.3)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())


# In[30]:


def cv(classifier, X_train, y_train):
    log("===============================================")
    classifier_name = str(type(classifier).__name__)
    now = time()
    log("Crossvalidating " + classifier_name + "...")
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    log("Crosvalidation completed in {0}s".format(time() - now))
    log("Accuracy: " + str(accuracy[0]))
    log("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    log("===============================================")
    return accuracy


# ## Add Extra Features

# To run the extra feature function we have to oad the data again because we have already removed the special charactrs and numbers.

# In[31]:


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data.rename(columns={'Category': 'emotion'}, inplace=True)
test_data.rename(columns={'Category': 'emotion'}, inplace=True)

train_data = train_data[train_data['emotion'] != 'Tweet']
test_data = test_data[test_data['emotion'] != 'Tweet']

Feature name 	Explanation
Number of uppercase 	people tend to express with either positive or negative emotions by using A LOT OF UPPERCASE WORDS
Number of ! 	exclamation marks are likely to increase the strength of opinion
Number of ? 	might distinguish neutral tweets - seeking for information
Number of positive emoticons 	positive emoji will most likely not occur in the negative tweets
Number of negative emoticons 	inverse to the one above
Number of … 	commonly used in commenting something
Number of quotations 	same as above
Number of mentions 	sometimes people put a lot of mentions on positive tweets, to share something good
Number of hashtags 	just for the experiment
Number of urls 	similiar to the number of mentions
# In[32]:


def add_extra_feature(df, tweet_column):
    
    # Print Number of Exclamation
    #length_of_excl = (len(re.findall(r'!', string)))
    df['number_of_exclamation'] = tweet_column.apply(lambda x: (len(re.findall(r'!', x))))
    
    # Number of ?
    #length_of_questionmark = (len(re.findall(r'?', string)))
    df['number_of_questionmark'] = tweet_column.apply(lambda x: (len(re.findall(r'[?]', x))))
    
    # Number of #
    df['number_of_hashtag'] = tweet_column.apply(lambda x: (len(re.findall(r'#', x))))
    
    # Number of @
    df['number_of_mention'] = tweet_column.apply(lambda x: (len(re.findall(r'@', x))))
    
    # Number of Quotes
    df['number_of_quotes'] = tweet_column.apply(lambda x: (len(re.findall(r"'", x))))

    # Number if underscore
    df['number_of_underscore'] = tweet_column.apply(lambda x: (len(re.findall(r'_', x))))
    
    
    #print((txt.split(" "), row))
    #print(row.split())


# In[33]:


# pass the train_data into add_extra_feature function
add_extra_feature(train_data, train_data["Tweet"])


# ## ADD EMOTICONS

# Here, users emoticons in a tweet also matters, so we will find the emoticons in a users tweet.

# In[34]:


## Emoticon Detector

class EmoticonDetector:
    emoticons = {}

    def __init__(self, emoticon_file="data/emoticons.txt"):
        from pathlib import Path
        content = Path(emoticon_file).read_text()
        positive = True
        for line in content.split("\n"):
            if "positive" in line.lower():
                positive = True
                continue
            elif "negative" in line.lower():
                positive = False
                continue

            self.emoticons[line] = positive

    def is_positive(self, emoticon):
        if emoticon in self.emoticons:
            return self.emoticons[emoticon]
        return False

    def is_emoticon(self, to_check):
        return to_check in self.emoticons


# In[35]:


ed = EmoticonDetector()


# In[36]:


processed_data = train_data.copy()

def add_column(column_name, column_content):
    processed_data.loc[:, column_name] = pd.Series(column_content, index=processed_data.index)

def count_by_lambda(expression, word_array):
    return len(list(filter(expression, word_array)))

add_column("splitted_text", map(lambda txt: txt.split(" "), processed_data["Tweet"]))

positive_emo = list(
    map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
        processed_data["splitted_text"]))
add_column("number_of_positive_emo", positive_emo)

negative_emo = list(map(
    lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
    processed_data["splitted_text"]))

add_column("number_of_negative_emo", negative_emo)


# In[37]:


train_data['number_of_positive_emo'] = positive_emo
train_data['number_of_negative_emo'] = negative_emo


# ## WHY EXTRA FEATURES
Let's see how (some) of the extra features distribute the data set. Some of them, i.e number exclamation marks, number of pos/neg emoticons do this surprisingly well. Notwithstanding the good severance, those features seldom transpire only on a small subset of the training dataset.
# In[38]:


sns.barplot(x='emotion', y='number_of_mention', data=train_data)
sns.despine()
plt.tight_layout()


# In[39]:


sns.barplot(x='emotion', y='number_of_negative_emo', data=train_data)
sns.despine()
plt.tight_layout()


# In[40]:


sns.barplot(x='emotion', y='number_of_questionmark', data=train_data)
sns.despine()
plt.tight_layout()


# In[41]:


sns.barplot(x='emotion', y='number_of_exclamation', data=train_data)
sns.despine()
plt.tight_layout()


# In[42]:


sns.barplot(x='emotion', y='number_of_positive_emo', data=train_data)
sns.despine()
plt.tight_layout()


# In[43]:


sns.barplot(x='emotion', y='number_of_hashtag', data=train_data)
sns.despine()
plt.tight_layout()


# In[44]:


sns.barplot(x='emotion', y='number_of_underscore', data=train_data)
sns.despine()
plt.tight_layout()


# ## Preapre training data for model

# In[45]:


train_data.head()


# In[46]:


# apply the clean tweet function
train_data['Tweet'] = train_data['Tweet'].apply(clean_tweets)


# In[47]:


## Tokenize data
train_data['text'] = train_data['Tweet'].apply(tokenize)
train_data['tokenized'] = train_data['text'].apply(stemming)


# In[48]:


## BAG OF WORDS
wordlist= []
if os.path.isfile("wordlist.csv"):
    word_df = pd.read_csv("wordlist.csv")
    word_df = word_df[word_df["occurrences"] > 3]
    wordlist = list(word_df.loc[:, "word"])

label_column = ["label"]
columns = label_column + list(map(lambda w: w + "_bow",wordlist))
labels = []
rows = []
for idx in train_data.index:
    current_row = []
        # add label
    current_label = train_data.loc[idx, "emotion"]
    labels.append(current_label)
    current_row.append(current_label)

    # add bag-of-words
    tokens = set(train_data.loc[idx, "text"])
    for _, word in enumerate(wordlist):
        current_row.append(1 if word in tokens else 0)

    rows.append(current_row)

data_model = pd.DataFrame(rows, columns=columns)
data_labels = pd.Series(labels)


# In[49]:


dat1 = train_data
dat2 = data_model

dat1 = dat1.reset_index(drop=True)
dat2 = dat2.reset_index(drop=True)

data_model = dat1.join(dat2)


# In[50]:


train_data.columns


# In[51]:


## Drop the columns in data_model
data_model = data_model.drop(columns=['emotion','Tweet','text', 'tokenized','Id'], axis=1)


# ## Experiment 2: Added feature + Random Forest

# In[52]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(random_state=seed,n_estimators=403,n_jobs=-1))


# In[53]:


rf_acc = cv(RandomForestClassifier(n_estimators=403,n_jobs=-1, random_state=seed),data_model.drop(columns='label',axis=1), data_model['label'])


# ## Experiment 3: Added Feature + XGBoost

# In[55]:


from xgboost import XGBClassifier as XGBoostClassifier


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier(seed=seed))


# ## Experiment 4: Added Feature + Naive Bayes

# In[57]:


X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())


# ## Test Data

# In[58]:


test_data.head()


# In[61]:


test_data.columns


# In[64]:


# remove the tweets which contains Not available
test_data = test_data.rename(columns={"emotion": "Tweet"})
test_data = test_data[test_data['Tweet'] != "Not Available"]


# In[ ]:



# Drop null values
test_data = test_data.dropna() 

# add extra features
add_extra_feature(test_data, test_data['Tweet'])

# Clean tweets
test_data['Tweet'] = test_data['Tweet'].apply(clean_tweets)

## Tokenize data
test_data['text'] = test_data['Tweet'].apply(tokenize)
test_data['tokenized'] = test_data['text'].apply(stemming)


# In[ ]:


# wordlist
word_list(test_data)


# In[ ]:


## BAG OF WORDS
wordlist= []
if os.path.isfile("wordlist.csv"):
    word_df = pd.read_csv("wordlist.csv")
    word_df = word_df[word_df["occurrences"] > 3]
    wordlist = list(word_df.loc[:, "word"])

label_column = ["label"]
columns = label_column + list(map(lambda w: w + "_bow",wordlist))
labels = []
rows = []
for idx in test_data.index:
    current_row = []
        # add label
    current_label = test_data.loc[idx, "Tweet"]
    labels.append(current_label)
    current_row.append(current_label)

    # add bag-of-words
    tokens = set(test_data.loc[idx, "text"])
    for _, word in enumerate(wordlist):
        current_row.append(1 if word in tokens else 0)

    rows.append(current_row)

data_model = pd.DataFrame(rows, columns=columns)
data_labels = pd.Series(labels)


# In[ ]:


dat1 = test_data
dat2 = data_model

dat1 = dat1.reset_index(drop=True)
dat2 = dat2.reset_index(drop=True)

data_model = dat1.join(dat2)


# In[ ]:


test_model = pd.DataFrame()
test_model['original_id'] = data_model['Id']


# In[ ]:


data_model = data_model.drop(columns=['Tweet','text', 'tokenized','Id'], axis=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RF = RandomForestClassifier(n_estimators=403,max_depth=10)


# In[ ]:


RF.fit(data_model.drop(columns='label',axis=1),data_model['label'])


# In[ ]:


predictions = RF.predict(data_model.drop(columns='label',axis=1))


# In[ ]:


results = pd.DataFrame([],columns=["Id","Category"])
results["Id"] = test_model["original_id"].astype("int64")
results["Category"] = predictions
results.to_csv("results_xgb.csv",index=False)


# ### Conclusion
# The increase of microblogging sites like Twitter offers an unparalleled opening to form and employ approaches & technologies that search and mine for sentiments. The work presented in this paper specifies an approach for sentiment analysis on Twitter data. To unseal the sentiment, we extracted the relevant data from the tweets, added the features.
# The overall tweet sentiment was then calculated using a model that presented in this report. This work is exploratory in nature and the prototype evaluated is a preliminary prototype. 
# The models showed that prediction of text sentiment is a non-trivial task for machine learning. A lot of preprocessing is needed just to be able to run an algorithm. The main problem for sentiment analysis is to craft the machine representation of the text. Simple bag-of-words was definitely not enough to obtain satisfying results, thus a lot of additional features were created basing on common sense (number of emoticons, exclamation marks, number of question mark etc). I think that a slight improvement in classification accuracy for the given training dataset could be developed, but since it included highly skewed data (small number of negative cases), the difference will be probably in the order of a few percents. The thing that could possibly enhance classification outcomes will be to add a lot of additional examples (increase training dataset), because given 5971 examples clearly do not include all sequence of words used, further - a lot of emotion-expressing information certainly is missing.

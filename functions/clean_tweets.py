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

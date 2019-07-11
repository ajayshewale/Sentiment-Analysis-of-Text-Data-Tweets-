def test():
    # remove the tweets which contains Not available
    test_data = test_data.rename(columns={"emotion": "Tweet"})
    test_data = test_data[test_data['Tweet'] != "Not Available"]


    # Drop null values
    test_data = test_data.dropna()

    # add extra features
    add_extra_feature(test_data, test_data['Tweet'])

    # Clean tweets
    test_data['Tweet'] = test_data['Tweet'].apply(clean_tweets)

    ## Tokenize data
    test_data['text'] = test_data['Tweet'].apply(tokenize)
    test_data['tokenized'] = test_data['text'].apply(stemming)

    # wordlist
    word_list(test_data)

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



    dat1 = test_data
    dat2 = data_model

    dat1 = dat1.reset_index(drop=True)
    dat2 = dat2.reset_index(drop=True)

    data_model = dat1.join(dat2)

    test_model = pd.DataFrame()
    test_model['original_id'] = data_model['Id']

    data_model = data_model.drop(columns=['Tweet','text', 'tokenized','Id'], axis=1)

    from sklearn.ensemble import RandomForestClassifier

    RF = RandomForestClassifier(n_estimators=403,max_depth=10)

    RF.fit(data_model.drop(columns='label',axis=1),data_model['label'])

    predictions = RF.predict(data_model.drop(columns='label',axis=1))

    results = pd.DataFrame([],columns=["Id","Category"])
    results["Id"] = test_model["original_id"].astype("int64")
    results["Category"] = predictions
    results.to_csv("results_xgb.csv",index=False)

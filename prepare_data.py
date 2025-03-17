from sklearn.model_selection import train_test_split


def clean_dataset(df, min_words=9, max_words=500, debug=False):

    current_size = len(df)

    df = df[df.review_translate.apply(lambda x: len(x.split(' '))) > min_words]
    if debug:
        print(f'Dropped {current_size-len(df)} short reviews')
        current_size = len(df)

    df = df[df.review_translate.apply(lambda x: len(x.split(' '))) < max_words]
    if debug:
        print(f'Dropped {current_size-len(df)} long reviews')

    return df


def train_test_eval_split(df, test_ratio=0.15, eval_ratio=0.1, seed=42, debug=False):
    train_ratio = 1 - (test_ratio + eval_ratio)

    train_eval_df, test_df = train_test_split(df,
                                              test_size=test_ratio,
                                              stratify=df[['dataset_name', 'rating']],
                                              random_state=seed)

    train_df, eval_df = train_test_split(train_eval_df,
                                         test_size=eval_ratio / (train_ratio + eval_ratio),
                                         # Adjust to maintain overall ratio
                                         stratify=train_eval_df[['dataset_name', 'rating']],
                                         random_state=seed)
    if debug:
        print(f'Train set size: {len(train_df)}')
        print(f'Eval set size: {len(eval_df)}')
        print(f'Test set size: {len(test_df)}')

    train_df = train_df[['review_translate', 'rating']]
    train_df.columns = ['text', 'label']

    test_df = test_df[['review_translate', 'rating']]
    test_df.columns = ['text', 'label']

    eval_df = eval_df[['review_translate', 'rating']]
    eval_df.columns = ['text', 'label']

    train_df.to_csv('train_reviews.csv', index=False)
    test_df.to_csv('test_reviews.csv', index=False)
    eval_df.to_csv('eval_reviews.csv', index=False)

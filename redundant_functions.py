'''
The functions below were largely used to try and compute NaN values. This is 
an area I believe an be improved upon so for the time being, these functions will
be set asside until time can be allocated to address them.

Each Function has a docstring that describes its intended use.

Note: No functions in this file are intended to be used here and will not run unless
used elsewhere and the appropriate libraries imported.
'''


def age_brackets(df):
    '''
    ARGS:    df - DataFrame

    RETURNS :df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Fits 'age' columns into 10 year bins to form a general age category
    '''
    bins = [18,25,35,45,55,65,75,85,95,105]
    labels = ['18-24','25-34','35-44','45-54','55-64','65-74','75-84','85-94','95-104']
    df['age_bracket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    return df

def income_brackets(df):
    '''
    ARGS:    df - DataFrame

    RETURNS: df - DataFrame
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Fits 'income' rows into 3 year bins to form a general income category
    '''
    bins = [10000,50000,90000,130000]
    labels = ['10k-49k','50k-89k','90k+']
    df['income_bracket'] = pd.cut(df['income'], bins=bins, labels=labels, right=False)

    return df

def pre_model_process(transcript_df, portfolio_df, profile_df):
    '''
    ARGS:
    transcript_df - Transcript Dataframe
    portfolio_df  - Portfolio Dataframe
    profile_df    - Profile Dataframe

    RETURNS:
    df            - Dataframe
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Gets the to-be-used Dataframe and processes it so that its ready for model
    use in computing NaN values

    '''
    part_df = transcript_df.merge(profile_df, on= ['person_id', 'person_index'], how = 'outer')
    model_df = part_df.merge(portfolio_df, on= ['offer_id', 'offer_index'], how = 'outer')

    model_df = process_df(model_df)

    #prep for modelling
    model_df.age.replace(118, np.nan, inplace = True)
    model_df['ordinal_ms'] = model_df['membership_start'].map(datetime.toordinal).to_frame()
    model_df.drop(['membership_start'], axis = 1, inplace = True)
    model_df = pd.get_dummies(model_df, columns = ['event'], drop_first = True)

    return model_df

def fill_age_nan(df, filename = 'age_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df          - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments age_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    '''
    if build_model == True:
        age_model = create_age_model(df, score = score_model)
    else:
        age_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_age'] = age_model.predict(pred_df).round()
    df['age'].fillna(df['pred_age'], inplace = True)
    df = age_brackets(df)

    return df

def fill_income_nan(df, filename = 'income_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)

    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments income_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''

    if build_model == True:
        income_model = create_income_model(df, score = score_model)
    else:
        income_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_income'] = income_model.predict(pred_df)
    df = income_brackets(df)
    df['income_bracket'].fillna(df['pred_income'], inplace = True)

    return df

def fill_gender_nan(df, filename = 'gender_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments gender_model for predicting missing genders in dataframe and returns
    DataFrame with added column of predicted genders.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Current Model Accuracy:
    Holdout Validation Accuracy: 70.19%
    KFolds Cross-Validation Accuracy: 74.93%
    Stratified K-fold Cross-Validation Accuracy: 74.20%
    '''
    if build_model == True:
        gender_model = create_gender_model(df, score = score_model)
    else:
        gender_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    df['pred_gender'] = gender_model.predict(pred_df)
    df['gender'].fillna(df['pred_gender'], inplace = True)

    return df

def model_scoring(model, X_test, y_test, y_pred, skfold = True):
    print(f"Validating {model}")

    results_HV = model.score(X_test, y_test)
    kfold = KFold(n_splits=10, random_state=42, shuffle = True)
    print("Holdout Validation Accuracy: %.2f%%" % (results_HV.mean()*100.0))

    results_kfold = cross_val_score(model, X_test, y_pred, cv=kfold)
    print("KFolds Cross-Validation Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
    if skfold == True:
        skfold = StratifiedKFold(n_splits=3, random_state=42, shuffle = True)
        results_skfold = cross_val_score(model, X, y, cv=skfold)
        print("Stratified K-fold Cross-Validation Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

def create_age_model(df, filename = 'age_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df         - DataFrame
    filename   - Disired filename for model
    rs         - int or RandomState instance, default = 42
                 Controls the shuffling applied to the data
    score      - Performs Holdout Validation, KFolds Cross-Validation and
                 Stratified K-fold Cross-Validation accuracy checks on the
                 model. Prints results. (default = False)
    RETURNS:
    df         - age_model (for age prediction)
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Builds a model to predict age from and saves it as 'filename' for use later
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Accuracies with Age using RandomForestRegressor:
    Holdout Validation Accuracy: 69.95%
    KFolds Cross-Validation Accuracy: 43.79%

    Accuracies with Age Brackets using RandomForestClassifier:
    Holdout Validation Accuracy: 44.84%
    KFolds Cross-Validation Accuracy: 43.63%
    '''

    X_data = df[df['age'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data
    y = df[df['age'].notna()]['age']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    age_model = RandomForestRegressor(random_state = rs)
    age_model.fit(X_train,y_train)
    pickle.dump(age_model, open(filename, 'wb'))
    y_pred = age_model.predict(X_test)

    if score == True:
        model_scoring(age_model, X_test, y_test, y_pred, skfold = False)
    return age_model

def create_income_model(df, filename = 'income_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    score       - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)

    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments income_model for predicting missing ages in dataframe and returns
    model for future use.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Predicting exact values with such limited data was proving pretty inaccurate.

    The most successful (so far) regression produced the following results:
    (using RandomForestRegressor, RandomState = 42)
    Holdout Validation Accuracy: 52.65%
    KFolds Cross-Validation Accuracy: 46.62%
    Stratified K-fold Cross-Validation Accuracy: 39.56%

    Placing the income into 3 bins for Low Income, Middle Income and High Income
    and using a classifier prediction accuracy increases:
    (using RandomForestClassifier, RandomState = 42)
    Holdout Validation Accuracy: 67.97%
    KFolds Cross-Validation Accuracy: 77.86%
    Stratified K-fold Cross-Validation Accuracy: 76.65%
    '''
    bins = [10000,50000,90000,130000]
    labels = ['low','medium','high']
    df['income_bracket'] = pd.cut(df['income'], bins=bins, labels=labels, right=False)

    X_data = df[df['income_bracket'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data
    y = df[df['income_bracket'].notna()]['income_bracket']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    income_model = RandomForestClassifier(random_state = rs)
    income_model.fit(X_train,y_train)
    pickle.dump(income_model, open('income_pred.sav', 'wb'))
    y_pred = income_model.predict(X_test)
    if score == True:
        model_scoring(income_model, X_test, y_test, y_pred)

    return income_model

def create_gender_model(df, filename = 'gender_pred.sav', rs = 42, score = False):
    '''
    ARGS:
    df         - DataFrame
    filename   - Disired filename for model
    rs         - int or RandomState instance, default = 42
                 Controls the shuffling applied to the data
    score      - Performs Holdout Validation, KFolds Cross-Validation and
                 Stratified K-fold Cross-Validation accuracy checks on the
                 model. Prints results. (default = False)
    RETURNS:
    df         - age_model (for age prediction)
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Builds a model to predict age from and saves it as 'filename' for use later
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Current Model Accuracy:
    Holdout Validation Accuracy: 70.19%
    KFolds Cross-Validation Accuracy: 74.93%
    Stratified K-fold Cross-Validation Accuracy: 74.20%
    '''

    X_data = df[df['gender'].notna()].copy()
    X_data = X_data.dropna(axis = 1)
    X_data = X_data[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    X = X_data

    y_data = df[df['gender'].notna()]['gender']
    y = y_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rs)

    gender_model = RandomForestClassifier(random_state = rs)
    gender_model.fit(X_train,y_train)
    y_pred = gender_model.predict(X_test)
    pickle.dump(gender_model, open(filename, 'wb'))

    if score == True:
        model_scoring(gender_model, X_test, y_test, y_pred)
    return gender_model

def fill_age_nan(df, filename = 'age_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df          - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments age_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    '''
    if build_model == True:
        age_model = create_age_model(df, score = score_model)
    else:
        age_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_age'] = age_model.predict(pred_df).round()
    df['age'].fillna(df['pred_age'], inplace = True)
    df = age_brackets(df)

    return df
def fill_income_nan(df, filename = 'income_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)

    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments income_model for predicting missing ages in dataframe and returns
    DataFrame with added column of predicted ages.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    '''

    if build_model == True:
        income_model = create_income_model(df, score = score_model)
    else:
        income_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()

    df['pred_income'] = income_model.predict(pred_df)
    df = income_brackets(df)
    df['income_bracket'].fillna(df['pred_income'], inplace = True)

    return df
def fill_gender_nan(df, filename = 'gender_pred.sav', rs = 42, build_model = False, score_model = False):
    '''
    ARGS:
    df          - DataFrame
    filename    - Filename of model
    rs          - int or RandomState instance, default = 42
                  Controls the shuffling applied to the data before applying the
                  split in traintestsplit().
    build_model - When False, function uses saved model. (default = False)
                  When True, function rebuilds model.
    score_model - Performs Holdout Validation, KFolds Cross-Validation and
                  Stratified K-fold Cross-Validation accuracy checks on the
                  model. Prints results. (default = False)
    RETURNS:
    df         - DataFrame
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Impliments gender_model for predicting missing genders in dataframe and returns
    DataFrame with added column of predicted genders.
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Current Model Accuracy:
    Holdout Validation Accuracy: 70.19%
    KFolds Cross-Validation Accuracy: 74.93%
    Stratified K-fold Cross-Validation Accuracy: 74.20%
    '''
    if build_model == True:
        gender_model = create_gender_model(df, score = score_model)
    else:
        gender_model = pickle.load(open(filename, 'rb'))

    pred_df = df[['days_elapsed', 'person_index', 'offer_index',
           'success', 'ordinal_ms', 'event_offer received',
           'event_transaction']].copy()
    df['pred_gender'] = gender_model.predict(pred_df)
    df['gender'].fillna(df['pred_gender'], inplace = True)

    return df

def clean_data(build_model = True, rs = 42, score = False, compute_nans = True):
    '''
    ARGS:
    build_models     - Choose whether to rebuild models using functions or load
                       from saved models.
    rs(random_state) - int or RandomState instance, default = 42. Controls the
                       shuffling applied to the data
    score            - Performs Holdout Validation, KFolds Cross-Validation and
                       Stratified K-fold Cross-Validation accuracy checks on the
                       model. Prints results. (default = False)
    compute_nans     - If True (default), missing values for Age, Gender and
                       income are computed.
                       If False, NaN values are dropped.


    RETURNS:
    transcript_df    - Transcript Dataframe
    portfolio_df     - Portfolio Dataframe
    profile_df       - Profile Dataframe

    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    Cleans base data and brings it all together into one DataFrame.

    Income, Age, and Gender NaN values are computed rather than dropped (for now)
    as they represent over 10% of the data set.

    The other NaN values can be easily filled with 0
    '''
    portfolio_df_dirty = pd.read_json('data/portfolio.json', lines = True)
    profile_df_dirty = pd.read_json('data/profile.json', lines = True)
    transcript_df_dirty = pd.read_json('data/transcript.json', lines = True)

    # Clean data
    profile_df = clean_profile_df(profile_df_dirty)
    transcript_df = clean_transcript_df(transcript_df_dirty)
    portfolio_df = clean_portfolio_df(portfolio_df_dirty)

    transcript_df, portfolio_df, profile_df  = id_simpify(transcript_df, portfolio_df, profile_df)
    full_df = pre_model_process(transcript_df, portfolio_df, profile_df)

    # Offer type
    full_df['offer_type'].fillna('transaction', inplace = True)

    f_list = ['days_elapsed', 'person_index', 'offer_index', 'ordinal_ms',
    'event_offer received','event_transaction']

    for f in f_list:
        full_df[f].fillna(0, inplace = True)

    if compute_nans == True:
        if build_model == True:
            print('Building Models')
            # merge DataFrames
            full_df = fill_age_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Age')
            full_df = fill_income_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Income')
            full_df = fill_gender_nan(full_df, rs, build_model = True, score_model = score)
            print('Finished Gender')
        else:
            # merge DataFrames
            full_df = fill_age_nan(full_df)
            # full_df = fill_age_nan(full_df, rs, build_model = True, score_model = True)
            print('Finished Age')
            full_df = fill_income_nan(full_df)
            print('Finished Income')
            full_df = fill_gender_nan(full_df)
            print('Finished Gender')
    else:
        full_df = full_df[full_df['gender'].notna()].copy()
    # The Rest
    for x in full_df.columns:
        try:
            full_df[x].fillna(0, inplace = True)
        except:
            continue
    print('Clean Completed')
    return full_df

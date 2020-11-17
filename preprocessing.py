import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder


def preprocessing_data():
    # Import CSV files and create dateframes
    DATA_PATH = r'C:\Users\dlafrance\Desktop\CEBD1260\project\\'
    file_names = glob.glob(DATA_PATH + "*.csv")
    main_df = pd.read_csv(file_names[1])
    sessions_df = pd.read_csv(file_names[0])

    # Lower case all strings in both files
    main_df = main_df.applymap(lambda x: x.lower() if type(x) == str else x)
    sessions_df = sessions_df.applymap(lambda x: x.lower() if type(x) == str else x)

    # Main file pre-processing
    # Remove NDF values from our target
    main_df = main_df[main_df['country_destination'] != 'ndf']

    # Convert date fields to datetime format and split
    main_df['date_account_created'] = pd.to_datetime(main_df['date_account_created'])
    main_df['date_first_booking'] = pd.to_datetime(main_df['date_first_booking'])
    main_df['timestamp_first_active'] = pd.to_datetime(main_df['timestamp_first_active'], format='%Y%m%d%H%M%S')

    main_df['account_created_year'] = main_df['date_account_created'].dt.year
    main_df['account_created_month'] = main_df['date_account_created'].dt.month
    main_df['account_created_day'] = main_df['date_account_created'].dt.day
    main_df['account_created_weekday'] = main_df['date_account_created'].dt.weekday

    main_df['first_booking_year'] = main_df['date_first_booking'].dt.year
    main_df['first_booking_month'] = main_df['date_first_booking'].dt.month
    main_df['first_booking_day'] = main_df['date_first_booking'].dt.day
    main_df['first_booking_weekday'] = main_df['date_first_booking'].dt.weekday

    main_df['first_active_year'] = main_df['timestamp_first_active'].dt.year
    main_df['first_active_month'] = main_df['timestamp_first_active'].dt.month
    main_df['first_active_day'] = main_df['timestamp_first_active'].dt.day
    main_df['first_active_weekday'] = main_df['timestamp_first_active'].dt.weekday
    main_df['first_active_hour'] = main_df['timestamp_first_active'].dt.hour

    # Create new fields for days difference between first booking and other date fields
    main_df['diff_booking_first_active'] = (main_df['date_first_booking'] - main_df['timestamp_first_active']).dt.days
    main_df['diff_booking_account_created'] = (main_df['date_first_booking'] - main_df['date_account_created']).dt.days

    main_df.drop(['date_account_created', 'date_first_booking', 'timestamp_first_active'], axis=1, inplace=True)

    # Process age field to remove outliers and fill NAs
    age_median = main_df['age'].median()
    main_df.loc[(main_df['age'] > 90) | (main_df['age'] < 18), 'age'] = age_median
    main_df['age'].fillna(main_df['age'].median(), inplace=True)
    main_df['age'] = main_df['age'].astype(np.int64)
    main_df['age_group'] = pd.cut(main_df['age'], [0, 24, 34, 44, 54, 64, 90],
                                  labels=['18_24', '25_34', '35_44', '45_54', '55_64', '65_above'])

    # Affiliate NAs
    main_df['first_affiliate_tracked'].fillna('unknown', inplace=True)

    cat_cols = []
    for f in main_df.columns.values:
        if f in ['id', 'country_destination', 'age_group']: continue
        if main_df[f].dtype == 'object':
            cat_cols.append(f)

    # Encode categorical fields
    for f in cat_cols:
        df = pd.get_dummies(main_df[f], prefix=f)
        main_df = pd.concat([main_df, df], axis=1)
        main_df.drop(f, axis=1, inplace=True)

    # Label encode countries and age_groups
    le = LabelEncoder()
    main_df['country_destination'] = le.fit_transform(main_df['country_destination'])
    main_df['age_group'] = le.fit_transform(main_df['age_group'])

    # Sessions_df pre-processing
    # Fill NAs
    sessions_df['action'].fillna('-unknown-', inplace=True)
    sessions_df['action_type'].fillna('-unknown-', inplace=True)
    sessions_df['action_detail'].fillna('-unknown-', inplace=True)

    # Cap the quantile of secs elapsed as many are way too high
    sessions_df = sessions_df[sessions_df['secs_elapsed'] < sessions_df['secs_elapsed'].quantile(.90)]

    # Drop users without IDs
    sessions_df.dropna(inplace=True)

    # Aggregate sessions data
    agg_dict = {
        'action': ['count', 'nunique'],
        'action_type': ['nunique'],
        'action_detail': ['nunique'],
        'device_type': ['nunique'],
        'secs_elapsed': ['sum', 'mean', 'median', 'min', 'max']
    }

    agg_df = sessions_df.groupby('user_id').agg(agg_dict)
    agg_df.columns = [f'sessions_{col[0]}_{col[1]}' for col in agg_df.columns.tolist()]

    # Joining files
    main_df = main_df.merge(agg_df, left_on='id', right_on='user_id', how='left')

    # Replace missing numeric values by mean
    c = main_df.select_dtypes(np.number).columns
    main_df[c] = main_df[c].fillna(main_df[c].median())

    return main_df
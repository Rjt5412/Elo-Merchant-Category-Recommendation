import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle

from statsmodels.stats.outliers_influence import variance_inflation_factor

import gc
import warnings
warnings.filterwarnings('ignore')
#plt.style.use('dark_background')


#https://www.kaggle.com/fabiendaniel/elo-world
#Function to load data into pandas and reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def impute_data(df, type='mean'):
#perform mean/mode imputation for data
	columns  = list(df.columns[df.isna().any()])
	if type == 'mode':
		for column in columns:
			df[column].fillna(df[column].mode()[0], inplace=True)
	else:
		for column in columns:
			df[column].fillna(df[column].mean(), inplace=True)
	return df


def get_test_features(test_file, train_file):
# Generate the features in test.csv file and return the test dataframe

	#Load the test file
	train = reduce_mem_usage(pd.read_csv(train_file))
	test = reduce_mem_usage(pd.read_csv(test_file))
	#Impute the missing values
	test = impute_data(test, type='mode')
	

	train['rare_datapoints'] = 0
	train.loc[train['target'] < -30, 'rare_datapoints'] = 1
	#Fatures using first_active_month
	test['first_active_month'] = pd.to_datetime(test['first_active_month'])
	test['quarter_first_active_month'] = test['first_active_month'].dt.quarter
	test['first_active_month_diff_from_today'] = (datetime.datetime.today() - test['first_active_month']).dt.days

	#MeanEncoding 
	for feature in ['feature_1', 'feature_2', 'feature_3']:
		rare_data_mean = train.groupby([feature])['rare_datapoints'].mean()
		train[feature] = train[feature].map(rare_data_mean)
		test[feature] = test[feature].map(rare_data_mean)

	#Combining the date and categorical features
	test['cat_time_feature1'] = test['first_active_month_diff_from_today'] * test['feature_1']
	test['cat_time_feature2'] = test['first_active_month_diff_from_today'] * test['feature_2']
	test['cat_time_feature3'] = test['first_active_month_diff_from_today'] * test['feature_3']

	test['cat_time_ratio1'] = test['feature_1'] / test['first_active_month_diff_from_today']
	test['cat_time_ratio2'] = test['feature_2'] / test['first_active_month_diff_from_today']
	test['cat_time_ratio3'] = test['feature_3'] / test['first_active_month_diff_from_today']	
	
	#Some more aggregation features
	test['feature_sum'] = test['feature_1'] + test['feature_2'] + test['feature_3']
	test['feature_mean'] = test['feature_sum']/3
	test['feature_max'] = test[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
	test['feature_min'] = test[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
	test['feature_std'] = test[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

	feature_cols = ['feature_1', 'feature_2', 'feature_3']
	for f in feature_cols:
	    test['days_' + f] = test['first_active_month_diff_from_today'] * test[f]
	    test['days_' + f + '_ratio'] = test[f] / test['first_active_month_diff_from_today']
	
	return test


#Helper Functions

# Get the day of the week for the purchase date
def get_weekday(data):
  return data.dt.dayofweek

# Return 1 if the purchase date is on a weekend
def is_weekend(day):
  if day == 5 or day == 6:
    return 1
  else:
    return 0

#Return day of purchase
def get_day(date_obj):
  return date_obj.dt.day

#Return week of the year of purchase
def get_week_of_year(date_obj):
  return date_obj.dt.weekofyear

#Return hour of purchase
def get_hour(date_obj):
  return date_obj.dt.hour

#Return month of purchase date
def get_purchase_month(data):
  return data.dt.month

#Return the month phase during the purchase
#Eg. Early in the month, mid-month or at the end of the month
def get_time_of_month(date):
  if date.day <=10:
    return "Early"
  elif date.day > 10 and date.day <= 20:
    return "Mid"
  else:
    return "End"
  

# Time of the day during purchase
#Eg. Morning, Afternoon, Evening, Night
def get_time_of_day(time):
  if time.hour >= 4 and time.hour < 12:
    return "Morning"
  elif time.hour >= 12 and time.hour < 17:
    return "Afternoon"
  elif time.hour >= 17 and time.hour < 22:
    return "Evening"
  else:
    return "Night"
    

# Returns 1 if the purchase was made on a holiday(Saturdays and sundays excluded)
# Google Search : list of holidays in brazil 2017 and 2018
def get_isholiday(date):
  holiday_list=[
            '01-01-17', '14-02-17', '28-08-17', '14-04-17', '16-04-17', '21-04-17',
            '01-05-17', '15-06-17', '07-09-17', '12-10-17', '02-11-17', '15-11-17', 
            '24-12-17', '25-12-17', '31-12-17',
            '01-01-18', '14-02-18', '28-08-18', '14-04-18', '16-04-18', '21-04-18',
            '01-05-18', '15-06-18', '07-09-18', '12-10-18', '02-11-18', '15-11-18', 
            '24-12-18', '25-12-18', '31-12-18'
  ]
  date = date.strftime(format='%d-%m-%y') 
  if date in holiday_list:
    return 1
  else:
    return 0


def get_purchase_date_features(transactions):
#Get the aggregated(on card_id) date features

	transactions = transactions[['card_id', 'purchase_date', 'month_lag']]
	transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'], format='%Y-%m-%d %H:%M:%S')
	
	transactions['weekday'] = get_weekday(transactions['purchase_date'])
	transactions['is_weekend'] = transactions['weekday'].apply(lambda day: is_weekend(day))
	transactions['purchase_month'] = get_purchase_month(transactions['purchase_date'])
	transactions['purchase_day'] = get_day(transactions['purchase_date'])
	transactions['week_of_year'] = get_week_of_year(transactions['purchase_date'])
	transactions['purchase_hour'] = get_hour(transactions['purchase_date'])
	transactions['purchase_on_holiday'] = transactions['purchase_date'].apply(lambda date_obj: get_isholiday(date_obj))
	transactions['purchase_date'] = transactions['purchase_date'].dt.date

	transactions['month_diff'] = ((datetime.date.today() - transactions['purchase_date']).dt.days)//30
	transactions['month_diff'] +=transactions['month_lag']
	del transactions['month_lag']

	transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])

	#Now aggregated based on card_id
	aggregations = {
	    
	    'is_weekend': ['sum', 'mean'],
	    'purchase_on_holiday': ['sum', 'mean'],
	    'weekday' : ['nunique', 'sum', 'mean'],
	    'purchase_hour': ['nunique', 'mean', 'min', 'max'],
	    'week_of_year': ['nunique', 'mean', 'min', 'max'],
	    'month_diff': ['sum', 'mean', 'min', 'max', 'var', 'skew'],
	    'purchase_day': ['nunique', 'sum', 'min'],
	    'purchase_date' : [np.ptp, 'min', 'max'],
	    'purchase_month' : ['sum', 'mean', 'nunique']

	}

	aggregated_date_features = historical_transactions.groupby('card_id').agg(aggregations)
	aggregated_date_features.columns = ['transactions_'+'_'.join(col).strip() for col in aggregated_date_features.columns.values]
	aggregated_date_features['transactions_purchase_date_ptp'] = aggregated_date_features['transactions_purchase_date_ptp'].dt.days

	aggregated_date_features['transactions_purchase_date_max_diff_now'] = (datetime.datetime.today() - aggregated_date_features['transactions_purchase_date_max']).dt.days
	aggregated_date_features['transactions_purchase_date_min_diff_now'] = (datetime.datetime.today() - aggregated_date_features['transactions_purchase_date_min']).dt.days
	del aggregated_date_features['transactions_purchase_date_ptp']

	return aggregated_date_features, transactions['month_diff']


def get_categorical_aggregations(transactions):
#Get categorical feature aggregations for transactions

	cat_aggregations = pd.DataFrame()
		for col in ['category_2', 'category_3']:
		  cat_aggregations[col + '_mean'] = transactions.groupby([col])['purchase_amount'].transform('mean')
		  cat_aggregations[col + '_min'] = transactions.groupby([col])['purchase_amount'].transform('min')
		  cat_aggregations[col + '_max'] = transactions.groupby([col])['purchase_amount'].transform('max')
		  cat_aggregations[col + '_sum'] = transactions.groupby([col])['purchase_amount'].transform('sum')

	cat_aggregations['card_id'] = transactions['card_id']
	cat_aggregations = cat_aggregations.groupby('card_id').mean()
	cat_aggregations.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
	return cat_aggregations


def get_transactions_features(transactions_file):
# Get transaction features from transactions files	
	transactions = reduce_mem_usage(pd.read_csv(transactions_file))
	transactions.replace([-np.inf, np.inf], np.nan, inplace=True)
	
	#Impute the missing values
	transations = impute_data(transactions, type='mode')

	#Trimming the feature values
	transations['purchase_amount'] = transations['purchase_amount'].apply(lambda x: min(x, 0.8))
	transations['installments'].replace([-1, 999], np.nan, inplace=True)
	transations['installments'].fillna(transations['installments'].mode()[0], inplace=True)
	transations['price'] = transations['purchase_amount'] / (transations['installments'] + 0.01)

	transations['authorized_flag'] = transations['authorized_flag'].map({'Y': 1, 'N': 0})
	transations['category_1'] = transations['category_1'].map({'Y': 1, 'N': 0})
	
	#Get purchase_date features
	transaction_features, month_diff = get_purchase_date_features(transactions)

	transations['duration'] = transations['purchase_amount']*month_diff
	transations['amount_month_ratio'] = transations['purchase_amount']/month_diff

	#Get Category Aggregated Features 
	cat_aggregations = get_categorical_aggregations(transations)
	
	transaction_features = transaction_features.merge(cat_aggregations, on='card_id', how='left')
	#Delete unwanted dataframes
	#del cat_aggregations
	#gc.collect()

	transations.drop(columns=['purchase_date'], axis=1, inplace=True)

	#Get Numerical Aggregated Features
	aggregations = {
	    
	    'authorized_flag' : ['sum', 'mean'],
	    'category_1' : ['sum', 'mean'],
	    'card_id': ['size'],
	    'city_id' : ['nunique'],
	    'state_id' : ['nunique'],
	    'subsector_id' : ['nunique'],
	    'merchant_category_id' : ['nunique'],
	    'merchant_id': ['nunique'],
	    'month_lag' : ['sum', 'mean', 'min', 'max', 'var'],
	    'duration': ['mean', 'min', 'mean', 'max', 'var', 'skew'],
	    'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
	    'installments' : ['sum', 'mean', 'min', 'max', 'var'],
	    'purchase_amount' : ['sum', 'mean', 'min', 'max', 'var'],
	    'price': ['sum', 'mean', 'min', 'max', 'var', 'skew']
    
	}

	aggregated_numerical_features = transations.groupby('card_id').agg(aggregations)
	aggregated_numerical_features.columns = ['transactions_'+'_'.join(col).strip() 
	                           for col in aggregated_numerical_features.columns.values]

	transaction_features = transaction_features.merge(aggregated_numerical_features, on='card_id', how='left')

	return transaction_features

def get_predictions(test_file, train_file, historical_transactions_file, new_transactions_file):
#Take the files as input and generate loyalty score predictions

	#test_file: test file path
	#transactions: transactions file path
	#new_transactions: new_merchant_transactions file path

	#Get processed test dataframe with all features generated from test file
	test = get_test_features(test_file, train_file)	

	#Get historical_transactions features and merge
	transaction_features = get_transactions_features(historical_transactions_file)
	test = test.merge(transaction_features, on='card_id', how='left')

	#Get new_merchant_transactions features and merge
	new_transactions = get_transactions_features(new_transactions_file)
	test = test.merge(transaction_features, on='card_id', how='left')

	#Adding additional features
	test['transactions_purchase_date_max'] = pd.to_datetime(test['transactions_purchase_date_max'])
	test['new_transactions_purchase_date_max'] = pd.to_datetime(test['new_transactions_purchase_date_max'])

	test['transactions_purchase_date_min'] = pd.to_datetime(test['transactions_purchase_date_min'])
	test['new_transactions_purchase_date_min'] = pd.to_datetime(test['new_transactions_purchase_date_min'])
	test['first_active_month'] = pd.to_datetime(test['first_active_month'])	


	test['purchase_date_diff'] = (test['transactions_purchase_date_max'] - test['transactions_purchase_date_min']).dt.days
	test['new_purchase_date_diff'] = (test['new_transactions_purchase_date_max'] - test['new_transactions_purchase_date_min']).dt.days

	test['purchase_date_average'] = (test['purchase_date_diff'])/test['transactions_card_id_size']
	test['new_purchase_date_average'] = (test['new_purchase_date_diff'])/test['new_transactions_card_id_size']

	test['purchase_date_diff_now'] = (datetime.datetime.today() - test['transactions_purchase_date_max']).dt.days
	test['new_purchase_date_diff_now'] = (datetime.datetime.today() - test['new_transactions_purchase_date_max']).dt.days

	test['purchase_date_diff_now_min'] = (datetime.datetime.today() - test['transactions_purchase_date_min']).dt.days
	test['new_purchase_date_diff_now_min'] = (datetime.datetime.today() - test['new_transactions_purchase_date_min']).dt.days

	test['first_buy'] = (test['transactions_purchase_date_min'] - test['first_active_month']).dt.days
	test['new_first_buy'] = (test['new_transactions_purchase_date_min'] - test['first_active_month']).dt.days

	train['last_buy'] = (train['transactions_purchase_date_max'] - train['first_active_month']).dt.days
	test['last_buy'] = (test['transactions_purchase_date_max'] - test['first_active_month']).dt.days
	test['new_last_buy'] = (test['new_transactions_purchase_date_max'] - test['first_active_month']).dt.days	

	test['transactions_purchase_date_max'] = test['transactions_purchase_date_max'].astype(np.int64) * 1e-9
	test['new_transactions_purchase_date_max'] = test['new_transactions_purchase_date_max'].astype(np.int64) * 1e-9
	test['transactions_purchase_date_min'] = test['transactions_purchase_date_min'].astype(np.int64) * 1e-9
	test['new_transactions_purchase_date_min'] = test['new_transactions_purchase_date_min'].astype(np.int64) * 1e-9


	test['card_id_total'] = test['new_transactions_card_id_size'] + test['transactions_card_id_size'] 
	test['card_id_ratio'] = test['new_transactions_card_id_size'] / test['transactions_card_id_size']

	test['purchase_amount_total'] = test['new_transactions_purchase_amount_sum'] + test['transactions_purchase_amount_sum']
	test['purchase_amount_mean'] = test['new_transactions_purchase_amount_mean'] + test['transactions_purchase_amount_mean']
	test['purchase_amount_max'] = test['new_transactions_purchase_amount_max'] + test['transactions_purchase_amount_max']
	test['purchase_amount_min'] = test['new_transactions_purchase_amount_min'] + test['transactions_purchase_amount_min']
	test['purchase_amount_ratio'] = test['new_transactions_purchase_amount_sum'] / test['transactions_purchase_amount_sum']

	
	test['month_diff_mean'] = test['new_transactions_month_diff_mean'] + test['transactions_month_diff_mean']
	test['month_diff_ratio'] = test['new_transactions_month_diff_mean'] / test['transactions_month_diff_mean']
	test['month_lag_mean'] = test['new_transactions_month_lag_mean'] + test['transactions_month_lag_mean']
	test['month_lag_max'] = test['new_transactions_month_lag_max'] + test['transactions_month_lag_max']
	test['month_lag_min'] = test['new_transactions_month_lag_min'] + test['transactions_month_lag_min']

	test['category_1_mean'] = test['new_transactions_category_1_mean'] + test['transactions_category_1_mean']
	test['category_1_sum'] = test['new_transactions_category_1_sum'] + test['transactions_category_1_sum']


	test['installments_mean'] = test['new_transactions_installments_mean'] + test['transactions_installments_mean']
	test['installments_total'] = test['new_transactions_installments_sum'] + test['transactions_installments_sum']
	test['installments_ratio'] = test['new_transactions_installments_sum'] / test['transactions_installments_sum']
	test['installments_max'] = test['new_transactions_installments_max'] + test['transactions_installments_max']
	test['installments_min'] = train['new_transactions_installments_min'] + train['transactions_installments_min']


	test['duration_mean'] = test['new_transactions_duration_mean'] + test['transactions_duration_mean']
	test['duration_max'] = test['new_transactions_duration_max'] + test['transactions_duration_max']
	test['duration_min'] = test['new_transactions_duration_min'] + test['transactions_duration_min']


	test['amount_month_ratio_mean'] = test['new_transactions_amount_month_ratio_mean'] + test['transactions_amount_month_ratio_mean']
	test['amount_month_ratio_min'] = test['new_transactions_amount_month_ratio_min'] + test['transactions_amount_month_ratio_min']
	test['amount_month_ratio_max'] = test['new_transactions_amount_month_ratio_max'] + test['transactions_amount_month_ratio_max']

	test['CLV'] = test['transactions_card_id_size'] * test['transactions_purchase_amount_sum'] / test['transactions_month_diff_mean']
	test['new_CLV'] = test['new_transactions_card_id_size'] * test['new_transactions_purchase_amount_sum'] / test['new_transactions_month_diff_mean']

	test.drop(columns=['first_active_month'], inplace=True)

	test = impute_data(test, type='mode')

	
	test_scores_1 = np.zeros(test.shape[0])
	test_scores_2 = np.zeros(test.shape[0])
	predictions = np.zeros(test.shape[0])

	# The reason for using all folds models is because the predictions from these models(trained on stratified and repeatedkfolds) 
	# are averaged and stacked in the final Bayesian Ridge model 

	#Load all the models and predict the score
	for i in range(1,6):
		filename = 'Goss_StratifiedKFold_Model/Goss_StratifiedKFold_Model_' + str(i) + '.pkl'
		with open(filename, 'rb') as f:
			clf = pickle.load(f)
		test_scores_1 += clf.predict(test) / 5

	for i in range(1,11):
		filename = 'Goss_Repeated_KFold_Models/Goss_RepeatedKFold_Model_' + str(i) +'.pkl'
		with open(filename, 'rb') as f:
			clf = pickle.load(f)
		test_scores_2 += clf.predict(test) / 10

	#Stack the predictions for Bayesian Model
	test_stack = np.vstack([test_scores_1, test_scores_2]).transpose()

	filename = 'Bayesian_Models/Bayesian_Ridge_Stack_Model_1.pkl'
	with open(filename, 'rb') as f:
		clf = pickle.load(f)
	
	predictions += clf.predict(test_stack)

	return predictions


predictions = get_predictions(
	test_file= 'Data/test.csv', 
	train_file = 'Data/train.csv',
	historical_transactions_file = 'Data/historical_transactions.csv',
	new_transactions_file = 'Data/new_merchant_transactions.csv'
	)
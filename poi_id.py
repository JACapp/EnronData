
# ## Enron Data POI Classifier
# ### Jo Anna Capp
#set working directory
import os
os.chdir('D:/Documents/Udacity/IntroMachineLearning/ud120projectsmaster/ud120projectsmaster/UdacityP5')

#import all packages and modules here
import sys
import pickle
sys.path.append("../tools/")
import pandas
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from ggplot import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

features_list = ['poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#print "There are ", len(data_dict.keys()), "executives of interest in the Enron dataset"
#number of pois
num_poi = 0
for dic in data_dict.values():
    if dic['poi'] == 1:
        num_poi += 1
#print "There are ", num_poi, "identified persons of interest within the dataset"
#print "Data Dictionary Keys:"
#print(data_dict.keys())
#data dictionary format
#print "A typical key:value list: ", data_dict["SKILLING JEFFREY K"]

#change dataset to pandas dataframe
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

#count number of NA values
df.replace(to_replace='NaN', value=numpy.nan, inplace=True)
#print "Number of NaN values for each feature:"
#print df.isnull().sum()
#print "Shape of the dataframe: ", df.shape

#replace missing values with 0
df.replace(to_replace=numpy.nan, value=0, inplace=True)
#drop email address column
del df['email_address']

#df.describe()

# #### Outlier Removal

df= df.drop(df.index[[data_dict.keys().index("TOTAL"), data_dict.keys().index("THE TRAVEL AGENCY IN THE PARK")]])
df.describe()

#identify keys for potential outliers
for key, value in data_dict.items():
    if value['from_poi_to_this_person'] != 'NaN' and value['from_poi_to_this_person'] > 500:
        print "Max from_poi_to_this_person: ", key

for key, value in data_dict.items():
    if value['from_this_person_to_poi'] != 'NaN' and value['from_this_person_to_poi'] > 500:
        print "Max from_this_person_to_poi: ", key

for key, value in data_dict.items():
    if value['from_messages'] != 'NaN' and value['from_messages'] > 14000:
        print "Max from_messages: ", key

# #### Checking financial features

#Checking total pay
df_totalPay = pandas.DataFrame()
df_totalPay['name'] = data_dict.keys()
df_totalPay['total'] = (
    df['bonus'] +
    df['director_fees'] +
    df['deferral_payments'] +
    df['deferred_income'] +
    df['loan_advances'] +
    df['long_term_incentive'] +
    df['expenses'] +
    df['other'] +
    df['salary']
)
df_totalPay['total_payments'] = df['total_payments']
df_totalPay['equals?'] = (df_totalPay['total'] == df_totalPay['total_payments'])
df_totalPay['poi'] = df['poi']
print numpy.sum(df_totalPay['equals?']), " out of ", len(df_totalPay)
print "Summed Totals Different than Total Payments"
print df_totalPay[df_totalPay['equals?'] == False]

#Checking total stock value
df_totalStock = pandas.DataFrame()
df_totalStock['name'] = data_dict.keys()
df_totalStock['total'] = (
    df['restricted_stock'] +
    df['exercised_stock_options'] +
    df['restricted_stock_deferred']
    )
df_totalStock['total_stock_value'] = df['total_stock_value']
df_totalStock['equals?'] = (df_totalStock['total'] ==df_totalStock['total_stock_value'])
df_totalStock['poi'] = df['poi']
print numpy.sum(df_totalPay['equals?']), " out of ", len(df_totalPay)
print "Summed Totals Different from Total Stock Value"
print df_totalStock[df_totalStock['equals?'] == False]

#check
print data_dict["BELFER ROBERT"]
print"/n"
print data_dict["BHATNAGAR SANJAY"]

#remove two outliers
df = df.drop(df.index[[data_dict.keys().index("BELFER ROBERT"), data_dict.keys().index("BHATNAGAR SANJAY")]])
df.describe()


# ### Feature Selection

#create new feature: ratio of messages involving POI/total
df['poi_email_ratio'] = (df['from_poi_to_this_person'] + df['from_this_person_to_poi']) / (df['from_messages'] + df['to_messages'])
df['poi_email_ratio'].replace(to_replace='NaN', value=0, inplace=True)

#create new feature: payments/total compensation ratio
df['payment_ratio'] = df['total_payments']/(df['total_stock_value'] + df['total_payments'])
df.describe()

df.replace(to_replace='NaN', value=0, inplace=True)

# #### Train/Test Split

#convert pandas df to pickled dictionary
#drop rows in index corresponding to df
employees = employees = pandas.Series(list(data_dict.keys()))
employees = employees.drop(employees.index[[24, 101, 104, 120]])

#print df.index.values
#create new feature list
new_features_list = df.columns.values

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

# create a dictionary from the dataframe
df_dict = df.to_dict('index')

# Store to my_dataset for easy export below.
my_dataset = df_dict
my_feature_list = ['poi','bonus', 'deferral_payments', 'deferred_income', 'director_fees',
 'exercised_stock_options', 'expenses', 'from_messages',
 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances',
 'long_term_incentive', 'other', 'restricted_stock',
 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
 'to_messages', 'total_payments', 'total_stock_value', 'poi_email_ratio',
 'payment_ratio']
# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Since this data is very inbalanced, I'm going to use a stratified shuffle split to split my training and testing data.
#SSS to split into train/test data (code borrowed from tester.py)
cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.75, random_state = 42)
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

# ### Create/tune classifiers

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#create a pipeline for analysis - Random Forest
scaler = preprocessing.StandardScaler()
select = SelectKBest()
pca = PCA()
feature_selection = FeatureUnion([('select', select), ('pca', pca)],
                    transformer_weights={'pca': 10})
clf_rf = RandomForestClassifier()

steps = [('scale', scaler),('feature_selection', feature_selection),
        ('random_forest', clf_rf)]

pipeline = sklearn.pipeline.Pipeline(steps)

#search for best parameters
parameters = dict(feature_selection__select__k=[5, 10, 20],
                 feature_selection__pca__n_components=[2, 5, 10],
                 random_forest__n_estimators=[25, 50, 100],
                 random_forest__min_samples_split=[1, 3, 5, 10])

clf = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print select.get_support()
print clf.best_params_

#pipeline.fit(features_train, labels_train)
#pred = pipeline.predict(features_test)
report = sklearn.metrics.classification_report(labels_test, pred)
print report


# In[26]:

#take selectKbest out of the pipeline to look at top features
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), my_feature_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print "K-best features:", results_list


# The random forest classifier appears to produce the best precision and recall scores. I further tuned this classifier by adjusting the parameters in the grid search and testing whether both feature selection and dimensionality reduction were needed to produce the highest precision and recall scores. They were.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)

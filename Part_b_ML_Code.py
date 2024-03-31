
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm


def ReplaceReg(X, Y):
    dfTemp = df.dropna()
    x = dfTemp[X]
    y = dfTemp[Y]
    reg = np.polyfit(x, y, deg=1)
    toPred = df[df[Y].isna()]
    predict = np.poly1d(reg)
    Predicted = predict(toPred.loc[:, X])
    return Predicted.tolist()


def changeToPrec(att):
    TopPrec = df[att].quantile(q=0.995)
    dfTemp = df
    toRep = dfTemp[dfTemp[att] > TopPrec]
    # print(toRep[att])
    toRep[att] = TopPrec
    # print(toRep[att])
    dfTemp[dfTemp[att] > TopPrec] = toRep
    return dfTemp[att]


def fillByFormola(Cholesterol=None, triglyceride=None, HDL=None, LDL=None):
    if Cholesterol is None:
        return triglyceride + HDL + LDL
    if triglyceride is None:
        return 5 * (Cholesterol - LDL - HDL)
    if HDL is None:
        return Cholesterol - 0.2 * triglyceride - LDL
    elif LDL is None:
        return Cholesterol - 0.2 * triglyceride - HDL


# help function to make Discretization for normal att
def normalDiscretization(Att):
    # print(df[Att].describe())
    Range1 = df[Att].quantile(q=0).astype('int').astype('str') + '-' + df[Att].quantile(q=0.25).astype('int').astype(
        'str')
    Range2 = df[Att].quantile(q=0.25).astype('int').astype('str') + '-' + df[Att].quantile(q=0.5).astype('int').astype(
        'str')
    Range3 = df[Att].quantile(q=0.5).astype('int').astype('str') + '-' + df[Att].quantile(q=0.75).astype('int').astype(
        'str')
    Range4 = df[Att].quantile(q=0.75).astype('int').astype('str') + '-' + df[Att].quantile(q=1).astype('int').astype(
        'str')
    df[Att] = pd.qcut(df[Att], q=[0, .25, .5, .75, 1.], labels=[Range1, Range2, Range3, Range4])
    # print(df[Att])


# ================  Reading data  ==================
pd.options.display.max_columns = None
df = pd.read_csv('Xy_train.csv', header=0, low_memory=False)
total_rows = len(df)

# # ===============  Exception handling ==============
dfTemp = df[(df['gender'] == 'F') & (df['weight(kg)'].isin([55, 60, 65]))]  # Neg value in waist(cm)
dfTemp = dfTemp['waist(cm)'].astype('float')
df['waist(cm)'] = df['waist(cm)'].replace(
    {'-33': dfTemp.mean()})

df['triglyceride'] = df['triglyceride'].replace(
    {'-35': fillByFormola(Cholesterol=197, LDL=121, HDL=57)})

dfTemp = df[df['waist(cm)'] != 'ok']  # String value in waist(cm)
dfTemp = dfTemp[(dfTemp['gender'] == 'M') & (dfTemp['weight(kg)'].isin([100, 95, 105]))]
dfTemp = dfTemp['waist(cm)'].astype('float')
df['waist(cm)'] = df['waist(cm)'].replace({'ok': dfTemp.mean()})
df['waist(cm)'] = df['waist(cm)'].astype('float')

df['serum creatinine'] = df['serum creatinine'].replace(
    {999: df['serum creatinine'].mean()})  # Serum creatinine unlogical value

df['serum creatinine'] = df['serum creatinine'].astype('float')
dfTemp = df
dfTemp = dfTemp[dfTemp['serum creatinine'] <= 1.9]
df = dfTemp

## String in urine protein - dealing by replacing yes with 1
df['Urine protein'] = df['Urine protein'].replace({'yes': 1})
df['Urine protein'] = df['Urine protein'].astype('float')
df['eyesight(left)'] = df['eyesight(left)'].replace({9.9: 1})
df['eyesight(right)'] = df['eyesight(right)'].replace({9.9: 1})
## Neg value in systolic
df['systolic'] = df['systolic'].replace({-102: 102})  # No other exception attributes- asssume error when inserting data

## Nulls Handling:
df = df.dropna(thresh=24)  # Droping rows where 3+ column are missing

df['height(cm)'] = df.groupby(['gender', 'weight(kg)'])['height(cm)'].transform(lambda x: x.fillna(x.mean()))
df['weight(kg)'] = df['weight(kg)'].astype('float')
df['weight(kg)'] = df['weight(kg)'].fillna(value=df['weight(kg)'].mean())
df['waist(cm)'] = df['waist(cm)'].astype('float')
df['waist(cm)'] = df['waist(cm)'].fillna(df['waist(cm)'].mean())
df['AST'] = df['AST'].fillna(value=df['AST'].mean())
df['serum creatinine)'] = df['serum creatinine'].fillna(value=df['serum creatinine'].mean())
df['fasting blood sugar'] = df['fasting blood sugar'].fillna(value=df['fasting blood sugar'].mean())
df['height(cm)'] = df['height(cm)'].fillna(value=df['height(cm)'].mean())

df['eyesight(left)'] = df['eyesight(left)'].fillna(1.2)
df['eyesight(right)'] = df['eyesight(right)'].fillna(df['eyesight(left)'])
df['hearing(right)'] = df['hearing(right)'].fillna(df['hearing(left)'])
df['tartar'] = df['tartar'].fillna(df['dental caries'])
df['Urine protein'] = df['Urine protein'].fillna(1)
df['oral'] = df['oral'].fillna('Y')

toReplace = df[df['systolic'].isna()]  # replacing systolic from relaxation
toReplace['systolic'] = ReplaceReg('relaxation', 'systolic')
df[df['systolic'].isna()] = toReplace

toReplace = df[df['relaxation'].isna()]  # replacing relaxation from systolic
toReplace['relaxation'] = ReplaceReg('systolic', 'relaxation')
df[df['relaxation'].isna()] = toReplace

toReplace = df[df['hemoglobin'].isna()]  # replacing hemoglobin from weight(kg)
toReplace['hemoglobin'] = ReplaceReg('weight(kg)', 'hemoglobin')
df[df['hemoglobin'].isna()] = toReplace

toReplace = df[df['ALT'].isna()]  # replacing ALT from AST
toReplace['ALT'] = ReplaceReg('AST', 'ALT')
df[df['ALT'].isna()] = toReplace

# fill from fourmola
tofill = df[df['Cholesterol'].isna()]
df['Cholesterol'] = df['Cholesterol'].fillna(
    fillByFormola(triglyceride=tofill['triglyceride'], HDL=tofill['HDL'], LDL=tofill['LDL']))

tofill = df[df['triglyceride'].isna()]
df['triglyceride'] = df['triglyceride'].fillna(
    fillByFormola(Cholesterol=tofill['Cholesterol'], HDL=tofill['HDL'], LDL=tofill['LDL']))

tofill = df[df['HDL'].isna()]
df['HDL'] = df['HDL'].fillna(
    fillByFormola(Cholesterol=tofill['Cholesterol'], LDL=tofill['LDL'], triglyceride=tofill['triglyceride']))

tofill = df[df['LDL'].isna()]
df['LDL'] = df['LDL'].fillna(
    fillByFormola(Cholesterol=tofill['Cholesterol'], triglyceride=tofill['triglyceride'], HDL=tofill['HDL']))

df['serum creatinine'] = changeToPrec('serum creatinine')
df['HDL'] = changeToPrec('HDL')
df['triglyceride'] = changeToPrec('triglyceride')
df['LDL'] = changeToPrec('LDL')
df['systolic'] = changeToPrec('systolic')
df['Gtp'] = changeToPrec('Gtp')
df['fasting blood sugar'] = changeToPrec('fasting blood sugar')

# new Variable
df['BMI'] = df['weight(kg)'] / (((df['height(cm)']) / 100) * ((df['height(cm)']) / 100))

#--------------------------Remove Variables-----------

df.drop(columns="oral", inplace=True, errors="ignore")
df.drop(columns="ID", inplace=True, errors="ignore")

# df.drop(columns="LDL", inplace=True, errors="ignore")
# df.drop(columns="HDL", inplace=True, errors="ignore")
# df.drop(columns="triglyceride", inplace=True, errors="ignore")
# df.drop(columns="height(cm)", inplace=True, errors="ignore")
# df.drop(columns="weight(kg)", inplace=True, errors="ignore")


#-------------------------------discretization- part A--------------
# normalDiscretization('height(cm)')
# normalDiscretization('weight(kg)')

# df['ALT'] = pd.cut(df['ALT'], bins=[0, 25, 30, 90, 2914],
#                    labels=['Borderline Low', 'Normal', 'Borderline high', 'High'])
# df['AST'] = pd.cut(df['AST'], bins=[0, 25, 30, 90, 1090],
#                    labels=['Borderline Low', 'Normal', 'Borderline high', 'High'])


# ContinousVariables = df[
#     ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar',
#      'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST','ALT', 'Gtp']]

# CategorialVariables = df['gender', 'hearing(left)', 'hearing(right)', 'dental caries']






# ============================================================================ Part B ================================

#======== DEFINE DATA
## Define Seed
randState = 123

## Define Data Types
#without droped variables
#df = df.astype(dtype={"gender":"category","age":"float", "waist(cm)":"float","eyesight(left)":"float", "eyesight(right)":"float", "hearing(left)":"category","hearing(right)":"category", "systolic":"float","relaxation":"float","fasting blood sugar":"float","Cholesterol":"float","hemoglobin":"float","Urine protein":"category", "serum creatinine":"float", "AST":"float","ALT":"float", "Gtp":"float", "dental caries":"category","tartar":"category","smoking":"category"})
#with variables
df = df.astype(dtype={"gender":"category","age":"float","height(cm)":"float", "weight(kg)":"float", "waist(cm)":"float","eyesight(left)":"float", "eyesight(right)":"float", "hearing(left)":"category","hearing(right)":"category", "systolic":"float","relaxation":"float","fasting blood sugar":"float","Cholesterol":"float","triglyceride":"float","HDL":"float","LDL":"float","hemoglobin":"float","Urine protein":"category", "serum creatinine":"float", "AST":"float","ALT":"float", "Gtp":"float", "dental caries":"category","tartar":"category","smoking":"category"})

## Split to X's and Y datasets
X = df.drop(['smoking'], 1)
y = df['smoking']
pd.value_counts(y)


# ============================================= Desicion Tree===========================
## Split Train Data To Train And Test
X_train_BeforeDummies, X_test_BeforeDummies, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=randState)
X_train = X_train_BeforeDummies.copy()
X_test = X_test_BeforeDummies.copy()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print("Train\n-----------\n", pd.value_counts(y_train)/y_train.shape[0])
print("\nTest\n-----------\n", pd.value_counts(y_test)/y_test.shape[0])

#Creating a normalized data set
from sklearn.preprocessing import StandardScaler, MinMaxScaler
standard_scaler = StandardScaler()
X_train_s = standard_scaler.fit_transform(X_train)
X_test_s = standard_scaler.fit_transform(X_test)


#Creating a balanced data set by under-sampling
from imblearn.under_sampling import RandomUnderSampler
# Define the sampler
sampler = RandomUnderSampler()
# Use the sampler to sample the dataset
X_Train_Balance, y_Train_Balance = sampler.fit_resample(X_train, y_train)

print(f"Train_Balance : {X_Train_Balance.shape[0]}")
print(f"Test Balance size: {X_test.shape[0]}")
print("Train Balance\n-----------\n", pd.value_counts(y_Train_Balance)/y_Train_Balance.shape[0])
print("\nTest Balance\n-----------\n", pd.value_counts(y_test)/y_test.shape[0])


#
# #========BaseLine======
# ## Define Decision Tree model
# model_DecisionTree = DecisionTreeClassifier(criterion='entropy',random_state=randState)
# ## Fit the Decision Tree model
# model_DecisionTree.fit(X_train, y_train)

# ##cheke The Accuracy
# #ON *train*
# print(f"Accuracy ON *train*: {accuracy_score(y_true=y_train, y_pred=model_DecisionTree.predict(X_train)):.4f}")
# #ON *test*
# print(f"Accuracy ON *test*: {accuracy_score(y_true=y_test, y_pred=model_DecisionTree.predict(X_test)):.4f}")

# #BaseLine Decision Tree plot
## plt.figure(figsize=(12, 10))
# plot_tree(model_DecisionTree, filled=True,   fontsize=10)
# plt.show()

# ##cheke f1
# #ON *train*
# print(f"f1 ON *train*: {f1_score(y_true=y_train, y_pred=model_DecisionTree.predict(X_train)):.4f}")
# #ON *test*
# print(f"f1 ON *test*: {f1_score(y_true=y_test, y_pred=model_DecisionTree.predict(X_test)):.4f}")





## ==========================================TUNED TREE=======


#Find Optimal Parameters Values by Grid Search
list_criterion = ['entropy', 'gini']
list_max_depth = np.arange(1, 26, 1)
list_max_features = ['sqrt', 'log2', None]


param_grid_DT = {'criterion': list_criterion,
                            'max_depth':list_max_depth,
                            'max_features': list_max_features

                 }
param_grid_DT.values()



def tune_max_depth_Parameters(Res_dataFrame_dt, dt_grid_search):
    res = pd.DataFrame()
    for param in list_max_depth:
        for i in range(Res_dataFrame_dt.shape[0]):
            if (Res_dataFrame_dt.iloc[i].params.get('max_depth') == param) and Res_dataFrame_dt.iloc[i].params.get('criterion') == dt_grid_search.best_params_['criterion'] and Res_dataFrame_dt.iloc[i].params.get('max_features') == dt_grid_search.best_params_['max_features']:
                        res = res.append({'max_depth': param,
                    'train_acc': Res_dataFrame_dt.iloc[i].mean_train_score,
                    'test_acc': Res_dataFrame_dt.iloc[i].mean_test_score}, ignore_index=True)
    plt.figure(figsize=(13, 4))
    plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
    plt.plot(res['max_depth'], res['test_acc'], marker='o', markersize=4)
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.title("Effect of max_depth on train and validation accuracy")
    plt.show()

def tune_criterion_Parameters(Res_dataFrame_dt, dt_grid_search):
    res = pd.DataFrame()
    for param in list_criterion:
        for i in range(Res_dataFrame_dt.shape[0]):
            if (Res_dataFrame_dt.iloc[i].params.get('criterion') == param) and Res_dataFrame_dt.iloc[i].params.get('max_depth') == dt_grid_search.best_params_['max_depth'] and Res_dataFrame_dt.iloc[i].params.get('max_features') == dt_grid_search.best_params_['max_features']:
                        res = res.append({'criterion': param,
                    'train_acc': Res_dataFrame_dt.iloc[i].mean_train_score,
                    'validation_acc': Res_dataFrame_dt.iloc[i].mean_test_score}, ignore_index=True)

    objects = (res['criterion'])
    y_pos = np.arange(len(objects))
    performance = res['validation_acc']
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.title('Effect of different criterions on train and validation accuracy ')
    plt.show()

def tune_max_features_Parameters(Res_dataFrame_dt, dt_grid_search):
    res = pd.DataFrame()
    for param in list_max_features:
        for i in range(Res_dataFrame_dt.shape[0]):
            if (Res_dataFrame_dt.iloc[i].params.get('max_features') == param) and Res_dataFrame_dt.iloc[i].params.get('criterion') == dt_grid_search.best_params_['criterion'] and Res_dataFrame_dt.iloc[i].params.get('max_depth') == dt_grid_search.best_params_['max_depth']:
                        res = res.append({'max_features': param,
                    'train_acc': Res_dataFrame_dt.iloc[i].mean_train_score,
                    'test_acc': Res_dataFrame_dt.iloc[i].mean_test_score}, ignore_index=True)
    plt.figure(figsize=(13, 4))
    plt.plot(res['max_features'], res['train_acc'], marker='o', markersize=4)
    plt.plot(res['max_features'], res['test_acc'], marker='o', markersize=4)
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.title("Effect of max_features on train and validation accuracy")
    plt.show()


### Feature Importance
def feature_importanc_fun(dt_best_model):
    feature_importance = dt_best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx], fontsize=8)
    plt.title('Feature Importance At Tuned Decision Tree')
    plt.show()


def grid_search_fun(x_dataTrain, y_dataTrain,x_dataTest, y_dataTest, scoringMethod):
    dt_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=randState),
                               param_grid=param_grid_DT,
                               refit=True,
                               cv=10, verbose=1,
                               return_train_score=True, scoring=scoringMethod)
    dt_grid_search.fit(x_dataTrain, y_dataTrain)
    Res_dataFrame_dt=pd.DataFrame(dt_grid_search.cv_results_).sort_values('mean_test_score', ascending= False)[['params', 'mean_test_score', 'mean_train_score']]
    print(Res_dataFrame_dt)

    dt_best_model = dt_grid_search.best_estimator_
    print(dt_grid_search.best_params_, '\n')
    print(dt_best_model.get_params(), '\n')
    ### Print  Score For Tuned Decision Tree

    # print("f1 on *Train* (Tuned tree): ", round(f1_score(y_dataTrain, dt_best_model.predict(x_dataTrain)), 4))
    # print("f1 on *Test* (Tuned tree): ", round(f1_score(y_dataTest, dt_best_model.predict(x_dataTest)), 4))

    print("accuracy on *Train* (Tuned tree): ",round(accuracy_score(y_dataTrain, dt_best_model.predict(x_dataTrain)), 4))
    print("accuracy on *Test* (Tuned tree): ",round(accuracy_score(y_dataTest, dt_best_model.predict(x_dataTest)), 4))

    tune_max_depth_Parameters(Res_dataFrame_dt, dt_grid_search)
    tune_criterion_Parameters(Res_dataFrame_dt, dt_grid_search)
    tune_max_features_Parameters(Res_dataFrame_dt, dt_grid_search)

    #BEST Decision Tree plot
    plt.figure(figsize=(2^15,2^15))
    plot_tree(dt_best_model, filled=True,feature_names=X_train.columns, fontsize=5, max_depth=4)
    plt.show()


    feature_importanc_fun(dt_best_model)


# #grid_search_fun(X_train, y_train,X_test, y_test, 'accuracy')
# grid_search_fun(X_train, y_train,X_test, y_test, 'f1')
# #grid_search_fun(X_train_s, y_train,X_test_s, y_test, 'accuracy')
# grid_search_fun(X_Train_Balance, y_Train_Balance,X_test, y_test, 'accuracy')

#confusion_matrix
cm=confusion_matrix(y_true=y_test, y_pred=dt_best_model.predict(X_test))
print(cm)
# Get the class labels
class_labels = ['Class 0', 'Class 1']

# Print the confusion matrix
print('Confusion Matrix:')
print(' '*9, 'Predicted')
print(' '*7, end='')
for j, col in enumerate(cm.T):
    print(f'{class_labels[j]:^7s}', end='')
print()
print(' '*5, 'Actual')
for i, row in enumerate(cm):
    print(f'{class_labels[i]:<7s}', end='')
    for j, col in enumerate(row):
        print(f'{col:7d}', end='')
    print()

    # Generate the classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred=dt_best_model.predict(X_test))
print(report)



# =================== MDP ===========================
## Split Train Data To Train And Test

# Default parameters
# Not normalized:
model_MLP_Non = MLPClassifier()
model_MLP_Non.fit(X_train, y_train)
### Train
print("Not normalized:")
print(f"Accuracy Train: {accuracy_score(y_true=y_train, y_pred=model_MLP_Non.predict(X_train)):.3f}")
### Test
print(f"Accuracy Test: {accuracy_score(y_true=y_test, y_pred=model_MLP_Non.predict(X_test)):.3f}")


## Normalized by stamdart scalar:
standard_scaler = StandardScaler()
X_train_s = standard_scaler.fit_transform(X_train)
X_test_s = standard_scaler.fit_transform(X_test)
#
model_MLP = MLPClassifier()
model_MLP.fit(X_train_s, y_train)
### Train
print("Normalized by standart scalar:")
print(f"Accuracy Train: {accuracy_score(y_true=y_train, y_pred=model_MLP.predict(X_train_s)):.3f}")
### Test
print(f"Accuracy Test: {accuracy_score(y_true=y_test, y_pred=model_MLP.predict(X_test_s)):.3f}")



## Normalized by MinMax
model_MLP_MinMax = MLPClassifier()
minmax_scaler = MinMaxScaler()
X_train_n = minmax_scaler.fit_transform(X_train)
X_test_n = minmax_scaler.fit_transform(X_test)
model_MLP_MinMax.fit(X_train_n, y_train)
### Train
print("Normalized by MinMax:")
print(f"Accuracy Train: {accuracy_score(y_true=y_train, y_pred=model_MLP_MinMax.predict(X_train_n)):.3f}")
### Test
print(f"Accuracy Test: {accuracy_score(y_true=y_test, y_pred=model_MLP_MinMax.predict(X_test_n)):.3f}")


## Grid Search

### hidden_layer_sizes
param_grid = {
'hidden_layer_sizes': [(30, 30, 30), (50, 50, 50), (100, 100, 100), (30, 30), (50, 50), (100, 100), (100,)],
'hidden_layer_sizes': [(150,), (150, 150), (200,)]
}
grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid, refit=True, cv=10, verbose=3)
grid_search.fit(X_train_s, y_train)
#### Train
results = pd.DataFrame(grid_search.cv_results_)
results2 = pd.DataFrame()
results2['hidden_layer_sizes'] = results['param_hidden_layer_sizes']
results2['mean_test_score'] = results['mean_test_score']
results2['rank_test_score'] = results['rank_test_score']
results2.sort_values(by=['rank_test_score'], ascending=True)

### activation
param_grid = {
    'hidden_layer_sizes': [(100, 100, 100)],
    'activation': ['logistic', 'relu', 'tanh'],
}
grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42),
                           param_grid=param_grid, refit=True, cv=10, verbose=3)
grid_search.fit(X_train_s, y_train)
#### Train
results = pd.DataFrame(grid_search.cv_results_)
results2 = pd.DataFrame()
results2['activation'] = results['param_activation']
results2['mean_test_score'] = results['mean_test_score']
results2['rank_test_score'] = results['rank_test_score']
results2.sort_values(by=['rank_test_score'], ascending=True)

## learning_rate_init
param_grid = {
    'hidden_layer_sizes': [(100, 100, 100)],
    'activation': ['tanh'],
    'learning_rate_init': [0.001, 0.003, 0.005]
}
grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=350), param_grid=param_grid, refit=True, cv=10, verbose=3)
grid_search.fit(X_train_s, y_train)
#### Train
results = pd.DataFrame(grid_search.cv_results_)
results2 = pd.DataFrame()
results2['activation'] = results['param_activation']
results2['mean_test_score'] = results['mean_test_score']
results2['rank_test_score'] = results['rank_test_score']
results2.sort_values(by=['rank_test_score'], ascending=True)

## max_iter
param_grid = {
     'hidden_layer_sizes': [(100, 100, 100)],
     'activation': ['tanh'],
    'learning_rate_init': [0.001],
     'max_iter': [15, 30, 15, 200, 300],
 }
grid_search = GridSearchCV(estimator=MLPClassifier(),
                           param_grid=param_grid, refit=True, cv=10, verbose=3)
grid_search.fit(X_train_s, y_train)
#### Train
results = pd.DataFrame(grid_search.cv_results_)
results2 = pd.DataFrame()
results2['activation'] = results['param_activation']
results2['mean_test_score'] = results['mean_test_score']
results2['rank_test_score'] = results['rank_test_score']

results2.sort_values(by=['rank_test_score'], ascending=True)
print(results2)
## Final configuration
model_MLP = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='tanh', max_iter=300, learning_rate_init=0.001)
model_MLP.fit(X_train_s, y_train)
### Train
print(f"Accuracy Train: {accuracy_score(y_true=y_train, y_pred=model_MLP.predict(X_train_s)):.3f}")
### Test
print(f"Accuracy Test: {accuracy_score(y_true=y_test, y_pred=model_MLP.predict(X_test_s)):.3f}")

#confusion_matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=y_test, y_pred=model_MLP.predict(X_test_s))
print(cm)

# Get the class labels
class_labels = ['Class 0', 'Class 1']

# Print the confusion matrix
print('Confusion Matrix:')
print(' '*9, 'Predicted')
print(' '*7, end='')
for j, col in enumerate(cm.T):
    print(f'{class_labels[j]:^7s}', end='')
print()
print(' '*5, 'Actual')
for i, row in enumerate(cm):
    print(f'{class_labels[i]:<7s}', end='')
    for j, col in enumerate(row):
        print(f'{col:7d}', end='')
    print()

    # Generate the classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred=model_MLP.predict(X_test_s))
print(report)


# ============================================================================ K means ================================
df.drop(columns="ID", inplace=True, errors="ignore")
df.drop(columns="oral", inplace=True, errors="ignore")
# df.drop(columns="Cholesterol", inplace=True, errors="ignore")

df.drop(columns="LDL", inplace=True, errors="ignore")
df.drop(columns="HDL", inplace=True, errors="ignore")
df.drop(columns="Triglyceride", inplace=True, errors="ignore")

df['BMI'] = df['weight(kg)'] / (((df['height(cm)']) / 100) * ((df['height(cm)']) / 100))
df.drop(columns="weight(kg)", inplace=True, errors="ignore")
df.drop(columns="height(cm)", inplace=True, errors="ignore")


randState = 123
df = df.astype(
    dtype={"gender": "category", "age": "float", "waist(cm)": "float",
           "eyesight(left)": "float", "eyesight(right)": "float",
           "hearing(left)": "category", "hearing(right)": "category",
           "systolic": "float", "relaxation": "float", "fasting blood sugar": "float",
           "Cholesterol": "float", "hemoglobin": "float",
           "Urine protein": "category", "serum creatinine": "float", "AST": "float", "ALT": "float", "Gtp": "float",
           "dental caries": "category", "tartar": "category", "smoking": "category"})

## Split to X's and Y datasets
y = df['smoking']
X = df.drop(['smoking'], 1)
X_train = X.copy()
y_train_set = y.copy()
X_train = pd.get_dummies(X_train)

## Normalized by standart scalar:
standard_scaler = StandardScaler()
X_train_s = standard_scaler.fit_transform(X_train)

# pca
pca_s = PCA(n_components=2)
pca_s = pca_s.fit(X_train_s)
print(pca_s.explained_variance_ratio_)
print(pca_s.explained_variance_ratio_.sum())

SMOKINGpca_s = pca_s.transform(X_train_s)
SMOKINGpca_s = pd.DataFrame(SMOKINGpca_s, columns=['PC1', 'PC2'])
y_train_set = y.reset_index()
SMOKINGpca_s = pd.concat([SMOKINGpca_s, y_train_set], axis=1)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_s)
kmeans.cluster_centers_
kmeans.predict(X_train_s)

SMOKINGpca_s['smoking'] = y_train_set['smoking']
SMOKINGpca_s['cluster'] = kmeans.predict(X_train_s)
SMOKINGpca_s

sns.scatterplot(x='PC1', y='PC2', hue='smoking', data=SMOKINGpca_s, palette='Accent')
plt.scatter(pca_s.transform(kmeans.cluster_centers_)[:, 0], pca_s.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.title('X_train colored by target, compared to cluster area')
plt.show()

from sklearn.metrics import accuracy_score
print("Train accuracy: ",round(accuracy_score(y_train_set['smoking'],y_pred= kmeans.predict(X_train_s)),3))

from sklearn.metrics import accuracy_score
def Selectedmean(K,  dbi_list, sil_list, iner_list, chi_list):
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X_train_s)
    kmeans.cluster_centers_
    predictKmeans = kmeans.predict(X_train_s)
    SMOKINGpca_s[f'K={K}'] = predictKmeans
    SMOKINGpca_s['smoking'] = y_train_set['smoking']
    SMOKINGpca_s['cluster'] = kmeans.predict(X_train_s)
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=SMOKINGpca_s, palette='Accent')
    plt.show()
    print("Train accuracy: ",round(accuracy_score(y_train_set['smoking'],y_pred= kmeans.predict(X_train_s)),3))
    iner = kmeans.inertia_
    sil = silhouette_score(X_train_s, predictKmeans)
    dbi = davies_bouldin_score(X_train_s, predictKmeans)
    labels = kmeans.labels_
    cal = metrics.calinski_harabasz_score(X_train_s, labels)
    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)
    chi_list.append(cal)

iner_list = []
dbi_list = []
sil_list = []
chi_list = []

def printList(title, plotName):
    plt.plot(range(2, 10, 1),plotName, marker='o')
    plt.title(f'{title}')
    plt.xlabel("Number of clusters")
    plt.show()
for i in range(2,10,1):
    Selectedmean(i, dbi_list, sil_list, iner_list, chi_list)
printList('Inertia', iner_list)
printList('Silhouette', sil_list)
printList('Calinski-Harabasz Index', chi_list)
printList('Davies-bouldin', dbi_list)


# Plot with backgroud
pca = PCA(n_components=2)
X_train = X_train_s.copy()
data_pca = pca.fit_transform(X_train)

data_pca = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
new_y_train = pd.DataFrame(y_train_set).reset_index()['smoking']
data_pca['target'] = new_y_train
kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=randState)
kmeans.fit(X_train)
data_pca['cluster'] = kmeans.predict(X_train)
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=data_pca)
plt.title('Clustering for X_train')
plt.show()

#### Visualisation
x_try = np.linspace(-5,15, 100)
y_try = np.linspace(-5, 15, 100)
predictions = pd.DataFrame()
for x in tqdm(x_try):
    for y in y_try:
        pred = kmeans.predict(pca.inverse_transform(np.array([x, y])).reshape(-1, 33))[0]
        predictions = predictions.append(dict(X1=x, X2=y, y=pred), ignore_index=True)
plt.scatter(x=predictions[predictions.y == 0]['X1'], y = predictions[predictions.y == 0]['X2'], c='ivory')
plt.scatter(x=predictions[predictions.y == 1]['X1'], y = predictions[predictions.y == 1]['X2'], c='powderblue')
sns.scatterplot(x='PC1', y='PC2', hue='smoking', data=SMOKINGpca_s)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.title('X_train colored by smoking, compared to cluster area')
plt.show()

plt.show()
#
#confusion_matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=new_y_train, y_pred=kmeans.predict(X_train_s))
print(cm)

# Get the class labels
class_labels = ['Class 0', 'Class 1']

# Print the confusion matrix
print('Confusion Matrix:')
print(' '*9, 'Predicted')
print(' '*7, end='')
for j, col in enumerate(cm.T):
    print(f'{class_labels[j]:^7s}', end='')
print()
print(' '*5, 'Actual')
for i, row in enumerate(cm):
    print(f'{class_labels[i]:<7s}', end='')
    for j, col in enumerate(row):
        print(f'{col:7d}', end='')
    print()

    # Generate the classification report
from sklearn.metrics import classification_report
report = classification_report(new_y_train, y_pred=kmeans.predict(X_train_s))
print(report)


################################### Agglomerative Clustering ###################################

model_AC = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
clustering_AC = model_AC.fit(X_train_s)
clustering_AC_predict = clustering_AC.labels_
SMOKINGpca_s['Agglomerative Clustering'] = clustering_AC_predict

sns.scatterplot(x='PC1', y='PC2', hue='Agglomerative Clustering', data=SMOKINGpca_s, palette={0:'orange', 1:'yellow', 2:'green', 3:'purple'}, s=100, )
print("Train accuracy: ",round(accuracy_score(y_train_set['smoking'],y_pred= SMOKINGpca_s['Agglomerative Clustering'])))

## predit
toPred_df = pd.read_csv('X_test.csv', header=0, low_memory=False)
toPred_df.columns[toPred_df.isnull().any()]
# toPred_df['BMI'] = toPred_df['weight(kg)'] / (((toPred_df['height(cm)']) / 100) * ((toPred_df['height(cm)']) / 100))
print(toPred_df.dropna(thresh=24))
toPred_df = toPred_df.dropna(thresh=24)
toPred_df['height(cm)'] = toPred_df['height(cm)'].fillna(toPred_df['height(cm)'].mean())
toPred_df['weight(kg)'] = toPred_df['weight(kg)'].fillna(toPred_df['weight(kg)'].mean())
toPred_df['eyesight(left)'] = toPred_df['eyesight(left)'].replace({9.9: 1})
toPred_df['eyesight(right)'] = toPred_df['eyesight(right)'].replace({9.9: 1})
toPred_df['systolic'] = toPred_df['systolic'].replace({-118: 118})
toPred_df['fasting blood sugar'] = toPred_df['fasting blood sugar'].fillna(toPred_df['fasting blood sugar'].mean())
toPred_df['serum creatinine)'] = toPred_df['serum creatinine'].fillna(value=toPred_df['serum creatinine'].mean())
toPred_df['Urine protein'] = toPred_df['Urine protein'].fillna(1)
toPred_df['Urine protein'] = toPred_df['Urine protein'].replace({'yes': 1})

toPred_df['ALT'] = toPred_df['ALT'].fillna(toPred_df['ALT'].mean())
toPred_df['ALT'] = toPred_df['ALT'].replace({-18: 18})
toPred_df['ALT'] = toPred_df['ALT'].replace({-18: 18})
toPred_df['tartar'] = toPred_df['tartar'].replace({'yes': 'Y'})
toPred_df['tartar'] = toPred_df['tartar'].replace({'no': 'N'})

# fill from fourmola
toPred_df['LDL'] = toPred_df['LDL'].replace({-190: 190})

toPred_fill_df = toPred_df[toPred_df['Cholesterol'].isna()]
toPred_df['Cholesterol'] = toPred_df['Cholesterol'].fillna(
    fillByFormola(triglyceride=toPred_fill_df['triglyceride'], HDL=toPred_fill_df['HDL'], LDL=toPred_fill_df['LDL']))

toPred_fill_df = toPred_df[toPred_df['triglyceride'].isna()]
toPred_df['triglyceride'] = toPred_df['triglyceride'].fillna(
    fillByFormola(Cholesterol=toPred_fill_df['Cholesterol'], HDL=toPred_fill_df['HDL'], LDL=toPred_fill_df['LDL']))

toPred_fill_df = toPred_df[toPred_df['HDL'].isna()]
toPred_df['HDL'] = toPred_df['HDL'].fillna(
    fillByFormola(Cholesterol=toPred_fill_df['Cholesterol'], LDL=toPred_fill_df['LDL'], triglyceride=toPred_fill_df['triglyceride']))

toPred_fill_df = toPred_df[toPred_df['LDL'].isna()]
toPred_df['LDL'] = toPred_df['LDL'].fillna(
    fillByFormola(Cholesterol=toPred_fill_df['Cholesterol'], triglyceride=toPred_fill_df['triglyceride'], HDL=toPred_fill_df['HDL']))

toPred_df['gender'] = toPred_df['gender'].replace({'ok': 'M'})
toPred_df['gender'] = toPred_df['gender'].fillna('F')
toPred_df['age'] = toPred_df['age'].fillna(toPred_df['age'].mean())
toPred_df['relaxation'] = toPred_df['relaxation'].fillna(toPred_df['relaxation'].mean())
toPred_df['AST'] = toPred_df['AST'].fillna(toPred_df['AST'].mean())
toPred_df['serum creatinine'] = toPred_df['serum creatinine'].fillna(toPred_df['serum creatinine'].mean())
toPred_df['hearing(right)'] = toPred_df['hearing(right)'].fillna(1)
toPred_df['hearing(left)'] = toPred_df['hearing(left)'].fillna(1)

toPred_df.drop(columns="oral", inplace=True, errors="ignore")
toPred_df.drop(columns="LDL", inplace=True, errors="ignore")
ID = toPred_df['ID']
toPred_df.drop(columns="ID", inplace=True, errors="ignore")
toPred_df.drop(columns="HDL", inplace=True, errors="ignore")
toPred_df.drop(columns="triglyceride", inplace=True, errors="ignore")
toPred_df['BMI'] = toPred_df['weight(kg)'] / (((toPred_df['height(cm)']) / 100) * ((toPred_df['height(cm)']) / 100))
toPred_df.drop(columns="height(cm)", inplace=True, errors="ignore")
toPred_df.drop(columns="weight(kg)", inplace=True, errors="ignore")

toPred_df = toPred_df.astype(dtype={"gender":"category","age":"float", "waist(cm)":"float","eyesight(left)":"float", "eyesight(right)":"float", "hearing(left)":"category","hearing(right)":"category", "systolic":"float","relaxation":"float","fasting blood sugar":"float","Cholesterol":"float","hemoglobin":"float","Urine protein":"category", "serum creatinine":"float", "AST":"float","ALT":"float", "Gtp":"float", "dental caries":"category","tartar":"category"})
toPred = pd.get_dummies(toPred_df)
# Default parameters

# # Not normalized:
# model_MLP_Non = MLPClassifier()
# model_MLP_Non.fit(X_train, y_train)
# ### Train
# print("Not normalized:")
# print(f"Accuracy Train: {accuracy_score(y_true=y_train, y_pred=model_MLP_Non.predict(X_train)):.3f}")
# ### Test
# print(f"Accuracy Test: {accuracy_score(y_true=y_test, y_pred=model_MLP_Non.predict(X_test)):.3f}")

## Normalized by stamdart scalar:
standard_scaler = StandardScaler()
toPred_s = standard_scaler.fit_transform(toPred)
toPred_s


#Extract to excel the predict by MLP
Predictiob = model_MLP.predict(toPred_s)
ans1 = pd.DataFrame()
ans1['index'] = ID
ans1['pred'] = Predictiob
ans1.to_excel(r'C:\Users\marienbe\Desktop\BGU\4th year\Sem A\Machine learning\Assingments\Part B\predNN.xlsx', index=False)
































































































































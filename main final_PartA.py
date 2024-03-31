import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, stats
import statsmodels.api as sm
import numpy as np
from scipy.stats import chi2_contingency

# ============= Functions  ==================


def printHistogrem(attributes, bin):
    sns.distplot(df[attributes], hist=False, kde=True)
    plt.hist(df[attributes], bins=bin, density=True, color='lightblue')
    plt.title(f'{attributes.capitalize()} histogram:', fontsize=20)
    plt.xlabel(attributes, fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.show()


def printPieChart(attribute, legend=None):
    dfAtt = df.groupby(attribute)
    countAtt = dfAtt.size()
    AttProb = countAtt.apply(lambda countAtt: countAtt / total_rows)
    AttLabel = df[attribute].unique()
    plt.title(f'{attribute.capitalize()} data Pie chart:')
    plt.pie(AttProb, labels=AttProb.index.array, colors=sns.color_palette('Set2')[0:len(AttLabel)], autopct='%.2f%%')
    if legend:
        plt.legend(legend, loc=3)
    plt.show()


def printCounterPlot(attribute):
    dfAtt = df.groupby(attribute)
    countAtt = dfAtt.size()
    AttProb = countAtt.apply(lambda countAtt: countAtt / total_rows)
    print(AttProb)
    # get plot
    AttPlot = sns.countplot(x=df[attribute], palette="Set2", order=df[attribute].value_counts(ascending=False).index)
    rel_values = df[attribute].value_counts(ascending=False, normalize=True).values * 100
    orderLabel = df[attribute].value_counts(ascending=False).values
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(orderLabel, rel_values)]
    AttPlot.bar_label(container=AttPlot.containers[0], labels=lbls, fontsize=7)
    plt.show()


def myfunc(X):
    return slope * X + intercept


def PrintScatterPlot(xText, yText, x, y, R, model):
    ax = plt.axes()
    ax.set_facecolor("whitesmoke")
    plt.grid(True, linewidth=0.3)
    plt.scatter(x, y, color='steelblue', s=8)
    plt.plot(x, model, color='firebrick')
    plt.title(f'{xText.capitalize()},{yText.capitalize()} Scatter plot:', fontsize=16)
    plt.xlabel(xText.capitalize(), fontsize=10, weight='bold')
    plt.ylabel(yText.capitalize(), fontsize=10, weight='bold')
    plt.text(55, 23.5, f' Correlation: {round(R, 2)}, R squered : {round(R ** 2, 2)} ', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.show()


def polynomial_Regression_2D(xAtt, yAtt, c1, c2, c3):
    x = df[xAtt].astype('float')
    y = df[yAtt]
    mymodel = np.poly1d(np.polyfit(x, y, 2))
    myline = np.linspace(c1, c2, c3)
    # plot
    plt.figure()
    plt.title(xAtt + ' EFFECT ON ' + yAtt)
    plt.plot(x, y, '.')
    plt.plot(myline, mymodel(myline), '-', label='2-degree polynom')
    plt.legend(loc='lower right')
    plt.xlabel(xAtt)
    plt.ylabel(yAtt)
    plt.show()


def printBoxPlot(att1, att2):
    dfTemp3 = df
    sns.boxplot(x=dfTemp3[att2], y=dfTemp3[att1], palette="Set2")
    plt.title(f'{att1} by {att2}')
    plt.xlabel(att2)
    plt.ylabel(att1)
    plt.show()


def printBarPlot(xAtt, yAtt):
    barPlot_Df = df[[xAtt, yAtt]]
    cross_tab_prop = pd.crosstab(index=barPlot_Df[xAtt],
                                 columns=barPlot_Df[yAtt],
                                 normalize="index")
    cross_tab_prop.plot(kind='bar',
                        stacked=True,
                        colormap='tab20',
                        figsize=(10, 6))
    plt.legend(loc="upper left", ncol=2, title=yAtt)
    plt.xlabel(xAtt)
    plt.ylabel("Proportion")
    plt.show()


def prinHistBysmoking(xAtt):
    sns.histplot(data=df, x=xAtt, hue='smoking', kde=True)
    plt.xlabel(xAtt)
    plt.ylabel('frequency')
    plt.title("The Connection between smoking to " + xAtt)
    plt.show()


def ReplaceReg(X, Y):
    dfTemp = df.dropna()
    x = dfTemp[X]
    y = dfTemp[Y]
    reg = np.polyfit(x, y, deg=1)
    toPred = df[df[Y].isna()]
    predict = np.poly1d(reg)
    Predicted = predict(toPred.loc[:, X])
    return Predicted.tolist()

#fill Value By Formola we Found
def fillByFormola(Cholesterol=None, triglyceride=None, HDL=None, LDL=None):
    if Cholesterol is None:
        return triglyceride + HDL + LDL
    if triglyceride is None:
        return 5 * (Cholesterol - LDL - HDL)
    if HDL is None:
        return Cholesterol - 0.2 * triglyceride - LDL
    elif LDL is None:
        return Cholesterol - 0.2 * triglyceride - HDL

#Replace outliers (bigger then 99.5th percentile) with the 99.5th percentile
def changeToPrec(att):
    TopPrec = df[att].quantile(q=0.995)
    dfTemp = df
    toRep = dfTemp[dfTemp[att] > TopPrec]
    print(toRep[att])
    toRep[att] = TopPrec
    print(toRep[att])
    dfTemp[dfTemp[att] > TopPrec] = toRep
    return dfTemp[att]


def normalTestByshapiro(Att):
    testSha = shapiro(df[Att])
    print(Att + ' P-VAL= ')
    print(testSha.pvalue)


# help function to make Discretization for normal att
def normalDiscretization(Att):
    print(df[Att].describe())
    Range1 = df[Att].quantile(q=0).astype('int').astype('str') + '-' + df[Att].quantile(q=0.25).astype('int').astype(
        'str')
    Range2 = df[Att].quantile(q=0.25).astype('int').astype('str') + '-' + df[Att].quantile(q=0.5).astype('int').astype(
        'str')
    Range3 = df[Att].quantile(q=0.5).astype('int').astype('str') + '-' + df[Att].quantile(q=0.75).astype('int').astype(
        'str')
    Range4 = df[Att].quantile(q=0.75).astype('int').astype('str') + '-' + df[Att].quantile(q=1).astype('int').astype(
        'str')
    df[Att] = pd.qcut(df[Att], q=[0, .25, .5, .75, 1.], labels=[Range1, Range2, Range3, Range4])
    print(df[Att])


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


#Replace outliers (bigger then 99.5th percentile) with the 99.5th percentile
df['serum creatinine'] = changeToPrec('serum creatinine')
df['HDL'] = changeToPrec('HDL')
df['triglyceride'] = changeToPrec('triglyceride')
df['LDL'] = changeToPrec('LDL')
df['systolic'] = changeToPrec('systolic')
df['Gtp'] = changeToPrec('Gtp')
df['fasting blood sugar'] = changeToPrec('fasting blood sugar')




# =============  Printing data  ==============================
# ==============  Pie chart  ================================
printPieChart('gender')
printPieChart('hearing(left)', ['1-Good hearing', '2-Bad hearing'])
printPieChart('hearing(right)', ['1-Good hearing', '2-Bad hearing'])
printPieChart('dental caries', ['0-No Caries', '1-Caries'])
printPieChart('tartar')
printPieChart('smoking')  # Target variable

# ==============  Histogrem   ================================


waistPl = sns.distplot(df['waist(cm)'], color='lightblue')
plt.show()
printHistogrem('Cholesterol', 20)
printHistogrem('triglyceride', 20)
printHistogrem('LDL', 20)
printHistogrem('height(cm)', 10)
printHistogrem('age', 10)
printHistogrem('weight(kg)', 10)
printHistogrem('eyesight(left)', 20)
printHistogrem('eyesight(right)', 20)
printHistogrem('systolic', 20)
printHistogrem('relaxation', 20)
printHistogrem('fasting blood sugar', 20)
printHistogrem('triglyceride', 20)
printHistogrem('HDL', 20)
printHistogrem('LDL', 20)
printHistogrem('hemoglobin', 20)
printHistogrem('serum creatinine', 20)
printHistogrem('AST', 20)
printHistogrem('ALT', 20)
printHistogrem('Gtp', 20)
#
printCounterPlot('Urine protein')
#
#

# =========================== Correlation ===================

# =============== Heatmap Of Continuous Variables ========
ContinousVariables = df[
    ['hemoglobin', 'serum creatinine', 'AST', 'ALT', 'Gtp', 'LDL', 'HDL', 'triglyceride', 'Cholesterol',
     'fasting blood sugar', 'relaxation', 'systolic', 'eyesight(right)', 'eyesight(left)', 'waist(cm)', 'weight(kg)',
     'height(cm)', 'age']]
sns.heatmap(ContinousVariables.corr(), annot=True, cmap='coolwarm', annot_kws={'fontsize': 8, 'fontweight': 'bold'})
plt.title('correlations between the continous variables')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8,)
plt.show()




# ==============  ScatterPlots   ============================

# === weight(kg) - waist(cm) ===
x = df['weight(kg)'].astype('float')
y = df['waist(cm)']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('weight(kg)', 'waist(cm)', x, y, r, mymodel)

# === height(cm) - weight ===
x = df['height(cm)'].astype('float')
y = df['weight(kg)']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('height(cm)', 'weight(kg)', x, y, r, mymodel)

# === AST - ALT ====
x = df['AST'].astype('float')
y = df['ALT']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('AST', 'ALT', x, y, r, mymodel)

# ===  systolic - relaxation: ===
x = df['systolic'].astype('float')
y = df['relaxation']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('systolic', 'relaxation', x, y, r, mymodel)

# === Cholesterol - LDL =====
x = df['Cholesterol'].astype('float')
y = df['LDL']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('Cholesterol', 'LDL', x, y, r, mymodel)

# === hemoglobin - weight(kg) ===
x = df['weight(kg)'].astype('float')
y = df['hemoglobin']
slope, intercept, r, p, std_err = stats.linregress(x, y)
mymodel = list(map(myfunc, x))
PrintScatterPlot('weight(kg)', 'hemoglobin', x, y, r, mymodel)

# ========== Spacial correlation ================

# === LDL - HDL - triglyceride ~ Cholesterol =====
x = df[['LDL', 'HDL', 'triglyceride']]
y = df['Cholesterol']
res2 = sm.OLS(endog=y, exog=x.assign(intercept=0)).fit()
print(res2.summary())
dfTemp = df
dfTemp = dfTemp[dfTemp['LDL'] < 350]
dfTemp = dfTemp[dfTemp['HDL'] < 350]
dfTemp = dfTemp[dfTemp['triglyceride'] < 400]
x2 = dfTemp[['LDL', 'HDL', 'triglyceride']]
y2 = dfTemp['Cholesterol']
res2 = sm.OLS(endog=y2, exog=x2.assign(intercept=0)).fit()
print(res2.summary())



#====dental caries ~ tartar
printBarPlot('dental caries', 'tartar')
# =chi ^2 tast
# contingency table
tartar_caries_Df = df[['tartar', 'dental caries']]
chisqt = pd.crosstab(index=tartar_caries_Df['dental caries'],
                     columns=tartar_caries_Df['tartar'], margins=True)
print(chisqt)
chi2_stat, p, dof, expected = chi2_contingency(chisqt)
print(f"chi2 statistic:     {chi2_stat:.5g}")
print(f"p-value:            {p:.5g}")
print(f"degrees of freedom: {dof}")



# ===serum creatinine  ~ Urine protein
dfTemp2 = df
dfTemp2 = dfTemp2[dfTemp2['serum creatinine'] <= 3]
sns.boxplot(x=dfTemp2['Urine protein'], y=dfTemp2['serum creatinine'], palette="Set3")
plt.title('serum creatinine by Urine protein')
plt.xlabel('Urine protein')
plt.ylabel('serum creatinine')
plt.show()

# === age ~ relaxation: =====
polynomial_Regression_2D('age', 'relaxation', 15, 90, 150)

# === age ~ systolic: =====
polynomial_Regression_2D('age', 'systolic', 15, 90, 240)

# === gender - =====
printBoxPlot('height(cm)', 'gender')
printBoxPlot('weight(kg)', 'gender')
printBoxPlot('waist(cm)', 'gender')
printBoxPlot('serum creatinine', 'gender')



#======smoking correlation
printBoxPlot('age', 'smoking')
printBoxPlot('height(cm)', 'smoking')
printBoxPlot('weight(kg)', 'smoking')
printBoxPlot('waist(cm)', 'smoking')
printBoxPlot('serum creatinine', 'smoking')
printBoxPlot('eyesight(left)', 'smoking')
printBoxPlot('eyesight(right)', 'smoking')
printBoxPlot('systolic', 'smoking')
printBoxPlot('relaxation', 'smoking')
printBoxPlot('fasting blood sugar', 'smoking')
printBoxPlot('HDL', 'smoking')
printBoxPlot('LDL', 'smoking')
printBoxPlot('hemoglobin', 'smoking')
printBoxPlot('serum creatinine', 'smoking')
printBoxPlot('AST', 'smoking')
printBoxPlot('ALT', 'smoking')
printBoxPlot('Urine protein', 'smoking')

printBarPlot('gender', 'smoking')
printBarPlot('hearing(left)', 'smoking')
printBarPlot('hearing(right)', 'smoking')
printBarPlot('dental caries', 'smoking')
printBarPlot('tartar', 'smoking')



# Variable Selection:

df.drop(columns="oral", inplace=True, errors="ignore")
df.drop(columns="ID", inplace=True, errors="ignore")

# new Variable
df['BMI'] = df['weight(kg)'] / (((df['height(cm)']) / 100) * ((df['height(cm)']) / 100))
print(df['BMI'])

# Discretization:
prinHistBysmoking('AST')
prinHistBysmoking('ALT')

df['ALT'] = pd.cut(df['ALT'], bins=[0, 25, 30, 90, 2914],
                   labels=['Borderline Low', 'Normal', 'Borderline high', 'High'])
df['AST'] = pd.cut(df['AST'], bins=[0, 25, 30, 90, 1090],
                   labels=['Borderline Low', 'Normal', 'Borderline high', 'High'])

normalDiscretization('height(cm)')
normalDiscretization('weight(kg)')
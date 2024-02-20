#import data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns
df=pd.read_csv(r'C:/Users/vip/Documents/heart/heart_disease.csv' )

#put original dataset in another datset 
ds=df.copy()
#print(ds)
#print the first 5 rows
df.head()

######preprocessing steps###############

#drop duplicate variable
ds.duplicated()
#This method returns a new DataFrame without duplicate rows from the DataFrame ds
ds.drop_duplicates()

#print before replace NUll with a mode or median
dd=ds.isnull().sum()
print(dd)


###replaces the NULL values with a mode###
#This line calculates the mode (most frequent value)
mode=ds['education'].mode()
#replace any missing values (NaN) in the "education" column with the mode
ds['education'].fillna(mode[0],inplace=True)


###replaces the NULL values with a median###
#replace any missing values (NaN) of the dataset with the median value of that column.
# rounded up to the nearest integer using the math.ceil() function
x=ds['cigsPerDay'].median()
x=math.ceil(x)
ds['cigsPerDay'].fillna(x,inplace=True)

x=ds['BPMeds'].median()
x=math.ceil(x)
ds['BPMeds'].fillna(x,inplace=True)

x=ds['BMI'].median()
x=math.ceil(x)
ds['BMI'].fillna(x,inplace=True)

x=ds['heartRate'].median()
x=math.ceil(x)
ds['heartRate'].fillna(x,inplace=True)

x=ds['glucose'].median()
x=math.ceil(x)
ds['glucose'].fillna(x,inplace=True)

x=ds['totChol'].median()
x=math.ceil(x)
ds['totChol'].fillna(x,inplace=True)

#print after replace null with a mode or median
dd=ds.isnull().sum()
print(dd)

#encoding "Convert categorical features into numerical representations"

#Import the LabelEncoder class from the sklearn.preprocessing
l1 = LabelEncoder()
l1.fit(ds['education'])
ds.education = l1.transform(ds.education)

l1 = LabelEncoder()
l1.fit(ds['prevalentStroke'])
ds.prevalentStroke = l1.transform(ds.prevalentStroke)

l1 = LabelEncoder()
l1.fit(ds['Gender'])
ds.Gender = l1.transform(ds.Gender)

#Rename the 'Heart_ stroke' column in the dataset ds to 'Heartstroke'
ds = ds.rename(columns={'Heart_ stroke': 'Heartstroke'})
l1 = LabelEncoder()
l1.fit(ds['Heartstroke'])
ds.Heartstroke = l1.transform(ds. Heartstroke) 

# outliers detection
plt.hist(ds['glucose'])
plt.show()
lowerLimit = ds ['glucose'].quantile(0.25)
lowerLimit
ds [ds['glucose']< lowerLimit]
upperLimit = ds ['glucose'].quantile(0.75)
upperLimit
ds [ds['glucose'] > upperLimit]
ds=ds[(ds['glucose'] > lowerLimit) & (ds['glucose']<upperLimit)]


lowerLimit = ds ['totChol'].quantile(0.25)
lowerLimit
ds [ds['totChol']< lowerLimit]
upperLimit = ds ['totChol'].quantile(0.75)
upperLimit
ds [ds['totChol'] > upperLimit]
ds=ds[(ds['totChol'] > lowerLimit) & (ds['totChol']<upperLimit)]


lowerLimit = ds ['sysBP'].quantile(0.25)
lowerLimit
ds [ds['sysBP']< lowerLimit]
upperLimit = ds ['sysBP'].quantile(0.75)
upperLimit
ds [ds['sysBP'] > upperLimit]
ds=ds[(ds['sysBP'] > lowerLimit) & (ds['sysBP']<upperLimit)]

lowerLimit = ds ['BMI'].quantile(0.25)
lowerLimit
ds [ds['BMI']< lowerLimit]
upperLimit = ds ['BMI'].quantile(0.75)
upperLimit
ds [ds['BMI'] > upperLimit]
ds=ds[(ds['BMI'] > lowerLimit) & (ds['BMI']<upperLimit)]


lowerLimit = ds ['diaBP'].quantile(0.25)
lowerLimit
ds [ds['diaBP']< lowerLimit]
upperLimit = ds ['diaBP'].quantile(0.75)
upperLimit
ds [ds['diaBP'] > upperLimit]
ds=ds[(ds['diaBP'] > lowerLimit) & (ds['diaBP']<upperLimit)]

####################  Statistics
#get describtion and info of the data after cleaning
ds.describe()
ds.info()

#some statistcs on data of three columns education,bmi,age
sde=ds['education'].std()
ev=ds['education'].var()
m=ds['education'].median()
mode=ds['education'].mode()
mean=ds['education'].mean()

print("Standerd Deviton of eduction :",sde)
print("variance of eduction :",ev)
print("Median of eduction :",m)
print("mode of eduction :",mode)
print("mean of eduction :",mean)

x=ds['BMI'].std()
print("standered devition of BMI:" ,x)
xm=ds['BMI'].var()
print("variance of BMI:" ,xm)
l=ds['BMI'].median()
print("Median of Bmi :",l)
mode1=ds['BMI'].mode()
print("mode of BMI :",mode1)
mean1=ds['BMI'].mean()
print("mean of BMI :",mean1)

k=ds['age'].std()
print("standered devition of age:" ,k)
u=ds['age'].var()
print("variance of age:" ,u)
f=ds['age'].median()
print("Median of age :",f)
mode2=ds['age'].mode()
print("mode of age :",mode2)
mean2=ds['age'].mean()
print("mean of age :",mean2)

################ Visualization
#Bar Chart
ds['Gender'].value_counts().plot(kind='bar')
plt.title("Number Of Male And Female")

plt.xlabel('Gender')
plt.ylabel('Count')

plt.show()

# histogram of cigar Per Day
plt.hist(ds['cigsPerDay'])
plt.title("cigar Per Day")
 
plt.show()

#Scatter Plot
my_pal = ("#23297a","#ff8c00")

plt.scatter(ds['age'], ds['totChol'] )
plt.title("Scatter Plot")
sns.scatterplot(x='age', y='totChol', hue='Heart_ stroke', palette=my_pal, data=df)

plt.show()

#Pie Chart
counts = ds['Heartstroke'].value_counts()
plt.pie(counts, labels=['No Heart Stroke', 'Heart Stroke'], autopct='%1.1f%%')
plt.title('Heart Stroke')
plt.show()


#Box Plot
plt.title('Heart Rate')
plt.boxplot(ds['heartRate'],vert=False)
plt.show()


#Box Plot
plt.title('Glucose')
plt.boxplot(ds['glucose'],vert=False)
plt.show()



#Box Plot
plt.title('BMI')
plt.boxplot(ds['BMI'],vert=False)
plt.show()
###############Classification################

#split a dataset into training and testing sets
#y is the target variable (output expected)
y = ds['Heartstroke'] 

#X is the input dataset without the target variable 'Heartstroke'. 
X = ds.drop(['Heartstroke'],axis=1)

#drop other columns from the dataset if needed
#X = X.drop(['Gender'],axis=1)
#X = X.drop(['age'],axis=1)

#Import the train_test_split function from the sklearn.model_selection module
##Split the dataset 75% for training and  25% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

##evaluate a Support Vector Machine (SVM) classifier

#Import the SVC class from the sklearn.svm module
from sklearn.svm import SVC
svm_modle = SVC()
#trains it on the training data
svm_modle.fit(X_train, y_train)
SVC()
#makes predictions on the testing data
ysvc_pred = svm_modle.predict(X_test) 
#evaluates the classifier's accuracy
accuracy =svm_modle.score(X_test, y_test) 
print("SVM Accuracy:",accuracy)


#evaluate a Decision Tree classifier

#Import the DecisionTreeClassifier class from the sklearn.tree module
from sklearn.tree import DecisionTreeClassifier
#creates a Decision Tree classifier with a maximum depth of 3
dt = DecisionTreeClassifier(max_depth=3)
#trains it on the training data
dt.fit(X_train, y_train)
DecisionTreeClassifier()
#makes predictions on the testing data
y_dt_pred = dt.predict(X_test)
#evaluates the classifier's accuracy
accuracy = dt.score(X_test, y_test)
print("Decision Tree Accuracy:" ,accuracy)

#evaluate a K-Nearest Neighbors (KNN)

#Import the KNeighborsClassifier class from the sklearn.neighbors module
from sklearn.neighbors import KNeighborsClassifier
clf_knn= KNeighborsClassifier()
#trains it on the training data
clf_knn.fit(X_train, y_train)
KNeighborsClassifier()
#makes predictions on the testing data
y_knn_pred = clf_knn.predict(X_test)
#evaluates the classifier's accuracy
accracy = clf_knn.score(X_test,y_test)
print("KNN Accuracy:",accracy)

#evaluate a Logistic Regression classifier

#Import the LogisticRegression class from the sklearn.linear_model module
from sklearn.linear_model import LogisticRegression
#Create an instance of the LogisticRegression class with a maximum number of iterations set to 3000
logist = LogisticRegression(max_iter=3000)
##trains it on the training data
logist.fit(X_train, y_train)
LogisticRegression()
#makes predictions on the testing data
accuracy = logist.score(X_test,y_test)
y_pred = logist.predict(X_test)
#evaluates the classifier's accuracy
accuracy = dt.score(X_test, y_test)
print("Logistic Regression Accuracy:",accuracy)

# #evaluate a Gaussian Naive Bayes classifie

# #Import the GaussianNB class from the sklearn.naive_bayes module
# from sklearn.naive_bayes import GaussianNB
# NB = GaussianNB()
# #trains it on the training data
# NB.fit(X_train, y_train)
# #makes predictions on the testing data
# pred= NB.predict(X_test)
# #evaluates the classifier's accuracy
# accuracy = NB.score(X_test, y_test)
# print("NaiveBayse Accuracy",accuracy)

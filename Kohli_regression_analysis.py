import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from plotnine import *
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv("Virat_Kohli_odi.csv")

# check for missing values
print("Number of missing values:\n", df.isnull().sum())

# remove rows with missing values
df = df.dropna()

# select the numeric columns
numeric_cols = ['Runs Scored', 'Minutes Batted', 'Balls Faced', 'Boundaries', 'Strike Rate']

# convert the columns to numeric values
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# calculate the mean, median, mode, standard deviation and variance of each column
for col in numeric_cols:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode().values[0]
    std = df[col].std()
    var = df[col].var()
    print("\nColumn:", col)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)
    print("Standard Deviation:", std)
    print("Variance:", var)
    print("")

# select the independent and dependent variables for simple linear regression
X1 = df[['Balls Faced']]
Y1 = df['Runs Scored']

# select the independent and dependent variables for multiple linear regression
X2 = df[['Minutes Batted', 'Balls Faced', 'Boundaries']]
Y2 = df['Runs Scored']


# split the data into training and testing sets
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=0)

# apply simple linear regression
lin_reg = LinearRegression()
lin_reg.fit(X1_train, Y1_train)
lin_reg_score = lin_reg.score(X1_test, Y1_test)
print("\nSimple Linear Regression Score:", lin_reg_score)

# apply multiple linear regression
multi_reg = LinearRegression()
multi_reg.fit(X2_train, Y2_train)
multi_reg_score = multi_reg.score(X2_test, Y2_test)
print("\nMultiple Linear Regression Score:", multi_reg_score)

# apply polynomial regression
poly = PolynomialFeatures(degree=3)
X2_poly = poly.fit_transform(X2_train)
poly_reg = LinearRegression()
poly_reg.fit(X2_poly, Y2_train)
poly_reg_score = poly_reg.score(poly.fit_transform(X2_test), Y2_test)
print("\nPolynomial Regression Score:", poly_reg_score)

# predict the outcomes for simple linear regression
sample_balls = 50
lin_reg_pred = lin_reg.predict([[sample_balls]])
print("\nSimple Linear Regression Prediction for Runs Scored:", lin_reg_pred)

# predict the outcomes for multiple linear regression
sample_minutes = 60
sample_balls = 50
sample_boundaries = 5
multi_reg_pred = multi_reg.predict([[sample_minutes, sample_balls, sample_boundaries]])
print("\nMultiple Linear Regression Prediction for Runs Scored:", multi_reg_pred)

#predict the outcomes for polynomial regression
poly_reg_pred = poly_reg.predict(poly.fit_transform([[sample_minutes, sample_balls, sample_boundaries]]))
print("\nPolynomial Regression Prediction for Runs Scored:", poly_reg_pred)

# Plotting the Runs Scored vs. Balls Faced
plt.scatter(df['Runs Scored'], df['Balls Faced'])
plt.title('Runs Scored vs. Balls Faced')
plt.xlabel('Runs Scored')
plt.ylabel('Balls Faced')
plt.show()

# Plotting the Runs Scored vs. Minutes Batted
plt.scatter(df['Runs Scored'], df['Minutes Batted'])
plt.title('Runs Scored vs. Minutes Batted')
plt.xlabel('Runs Scored')
plt.ylabel('Minutes Batted')
plt.show()


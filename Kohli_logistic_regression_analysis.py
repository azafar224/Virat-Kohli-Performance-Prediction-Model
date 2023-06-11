import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the dataset
df = pd.read_csv("Virat_Kohli_odi.csv")

#modify the 'Result' column to binary variable
df['Result'] = df['Result'].apply(lambda x: 1 if x == 'w' else 0)

#select the independent and dependent variables
X = df[['Runs Scored', 'Minutes Batted', 'Balls Faced', 'Boundaries', 'Strike Rate']]
Y = df['Result']

#split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#apply logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

#predict the outcomes
Y_pred = log_reg.predict(X_test)

#calculate the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)

# print whether the result is a win or a loss
for i in range(len(Y_pred)):
    if Y_pred[i] == 1:
        print("Win")
    else:
        print("Loss")

print("Logistic Regression Accuracy:", accuracy)

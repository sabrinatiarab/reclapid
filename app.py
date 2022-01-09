import pandas as pd
import pickle
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'D:\Belajar\UAS Apro II\laptop.csv', sep=";")

X = df.iloc[:, :-1].values
Y = df.iloc[:, 7].values

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.05, random_state=100)

model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

y_pred_train = model_reg.predict(X_train)

y_pred_test = model_reg.predict(X_test)
print(r2_score(y_test, y_pred_test)*100)

X1 = sm.add_constant(X_train)
result = sm.OLS(y_train, X1).fit()
print(result.summary())

pickle.dump(model_reg, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

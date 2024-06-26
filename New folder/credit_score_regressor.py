import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

"""dropping useless data"""

df = pd.read_csv("CreditPrediction.csv")

df.drop_duplicates(inplace=True)

df.drop(columns='Marital_Status', inplace=True)
df.drop(columns='CLIENTNUM', inplace=True)
df.drop(columns=['Unnamed: 19'], inplace=True)

"""making all datas numerical"""

df['Card_Category'] = df['Card_Category'].replace({'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4})
df['Income_Category'] = df['Income_Category'].replace({'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3,
                                                       '$80K - $120K': 4, '$120K +': 5, 'Unknown': np.nan})
df['Education_Level'] = df['Education_Level'].replace({'Uneducated': 1, 'High School': 2, 'College': 3, 'Graduate': 4,
                                                       'Post-Graduate': 5, 'Doctorate': 6,
                                                       'Unknown': np.nan})
df = pd.get_dummies(df, columns=['Gender'])
df.reset_index(level=None, drop=True, inplace=True)

"""imputing nan data"""

knn_imputer = KNNImputer(n_neighbors=4)
imputed_data = knn_imputer.fit_transform(df)
imputed_data = pd.DataFrame(imputed_data, columns=df.columns)

"""removing outliers"""

Q1 = imputed_data.quantile(0.25)
Q3 = imputed_data.quantile(0.75)

IQR = Q3 - Q1

lower_fence = Q1 - (1.5 * IQR)
upper_fence = Q3 + (1.5 * IQR)


iqr_data = imputed_data[~((imputed_data < lower_fence) | (imputed_data > upper_fence)).any(axis=1)]

"""splitting the data into train and test"""

data = iqr_data.drop('Credit_Limit', axis=1)
target = iqr_data['Credit_Limit']
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
x_train.reset_index(level=None, drop=True, inplace=True)
y_train.reset_index(level=None, drop=True, inplace=True)
x_test.reset_index(level=None, drop=True, inplace=True)
y_test.reset_index(level=None, drop=True, inplace=True)

"""scaling the train and data


"""

scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(x_train)
normalized_train = pd.DataFrame(scaled_features, columns=[x_train.columns])
scaled_features = scaler.transform(x_test)
normalized_test = pd.DataFrame(scaled_features, columns=[x_test.columns])

"""implementing the random forest regressor"""

model = RandomForestRegressor(n_estimators=100, max_depth=11, random_state=42)
model.fit(normalized_train, y_train)
y_pred = model.predict(normalized_test)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(R2)
print(MSE)

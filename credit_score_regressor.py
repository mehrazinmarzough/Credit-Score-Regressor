import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer


df = pd.read_csv("CreditPrediction.csv")
df.drop_duplicates(inplace=True)
df.drop(columns=['Unnamed: 19'], inplace=True)
df['Marital_Status'] = df['Marital_Status'].fillna("Unknown")
df['Gender'] = df['Gender'].fillna("U")

df['Card_Category'] = df['Card_Category'].replace({'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4})
df['Income_Category'] = df['Income_Category'].replace({'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3,
                                                       '$80K - $120K': 4, '$120K +': 5, 'Unknown': None})
df['Education_Level'] = df['Education_Level'].replace({'Uneducated': 1, 'High School': 2, 'College': 3, 'Graduate': 4,
                                                       'Post-Graduate': 5, 'Doctorate': 6,
                                                       'Unknown': None})
df = pd.get_dummies(df, columns=['Gender'])
df = pd.get_dummies(df, columns=['Marital_Status'])

df.reset_index(level=None, drop=True, inplace=True)

data = df.drop('Credit_Limit', axis=1)
target = df['Credit_Limit']
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
x_train.reset_index(level=None, drop=True, inplace=True)
y_train.reset_index(level=None, drop=True, inplace=True)
x_test.reset_index(level=None, drop=True, inplace=True)
y_test.reset_index(level=None, drop=True, inplace=True)

features = x_train
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(features)
normalized_df = pd.DataFrame(scaled_features, columns=[x_train.columns])
normalized_df['Credit_Limit'] = y_train

knn_imputer = KNNImputer(n_neighbors=5)
numerical_cols = normalized_df.select_dtypes(include=['number']).columns
imputed_data = knn_imputer.fit_transform(normalized_df[numerical_cols])
imputed_x_train = pd.DataFrame(imputed_data, columns=numerical_cols)
print(imputed_x_train.isna().sum())

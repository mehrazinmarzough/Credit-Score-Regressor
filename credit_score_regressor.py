import pandas as pd

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

print(df.info())

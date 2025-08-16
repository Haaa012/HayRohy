import pandas as pd
import joblib

df = pd.read_csv('./data/cleaned_data.csv')
print(df.head())

model = joblib.load('./models/model.pkl')
sample = df.iloc[:1, :-1]  # Toutes les colonnes sauf la cible
print("Prediction:", model.predict(sample))

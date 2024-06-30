import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Charger les données
data = pd.read_excel('Insurance-data.xlsx')

# Pré-traiter les données
scaler = StandardScaler()
encoder_sex = LabelEncoder()
encoder_smoker = LabelEncoder()
encoder_region = LabelEncoder()

# Ajuster et transformer les données d'entraînement
data[['age', 'bmi', 'children']] = scaler.fit_transform(data[['age', 'bmi', 'children']])
data['sex'] = encoder_sex.fit_transform(data['sex'])
data['smoker'] = encoder_smoker.fit_transform(data['smoker'])
data['region'] = encoder_region.fit_transform(data['region'])

# Séparer les features et la cible
X = data.drop(columns=['charges'])
y = data['charges']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Créer le dossier models s'il n'existe pas
if not os.path.exists('models'):
    os.makedirs('models')

# Sauvegarder le modèle et les objets scaler et encoder
with open('models/model_random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/encoder_sex.pkl', 'wb') as f:
    pickle.dump(encoder_sex, f)

with open('models/encoder_smoker.pkl', 'wb') as f:
    pickle.dump(encoder_smoker, f)

with open('models/encoder_region.pkl', 'wb') as f:
    pickle.dump(encoder_region, f)

print("Modèle et objets sauvegardés avec succès.")

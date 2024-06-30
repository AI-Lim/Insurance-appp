import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px

# Charger le modèle et les objets scaler et encoder ajustés
with open('models/model_random_forest.pkl', 'rb') as f:
    model_random_forest = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoder_sex.pkl', 'rb') as f:
    encoder_sex = pickle.load(f)

with open('models/encoder_smoker.pkl', 'rb') as f:
    encoder_smoker = pickle.load(f)

with open('models/encoder_region.pkl', 'rb') as f:
    encoder_region = pickle.load(f)

# Fonction pour pré-traiter les données
def preprocess_data(data):
    data[['age', 'bmi', 'children']] = scaler.transform(data[['age', 'bmi', 'children']])
    data['sex'] = encoder_sex.transform(data['sex'])
    data['smoker'] = encoder_smoker.transform(data['smoker'])
    data['region'] = encoder_region.transform(data['region'])
    return data

# Fonction pour prédire les charges
def predict(data):
    data = preprocess_data(data)
    prediction = model_random_forest.predict(data)
    return prediction

# Initialiser la liste des prédictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = []

# Initialiser l'état de la page actuelle
if 'page' not in st.session_state:
    st.session_state['page'] = "Welcome"

# Fonction pour changer de page
def set_page(page):
    st.session_state['page'] = page

# Configurer la mise en page
st.set_page_config(layout="wide")

# Barre latérale de navigation
st.sidebar.image("logo.png", width=150)
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Welcome", "Home", "View Data"], index=["Welcome", "Home", "View Data"].index(st.session_state['page']), on_change=lambda: set_page(menu))

# Page Welcome
if st.session_state['page'] == "Welcome":
    st.title('Welcome to the Insurance Predict App')
    st.write("### This application helps you predict insurance charges based on your personal details.")
    
    # Utiliser du HTML pour gérer la navigation
    if st.button('Get Started'):
        set_page("Home")

# Page Home
elif st.session_state['page'] == "Home":
    st.title('Insurance Predict App')
    st.write("### Enter your personal details to predict the insurance charges.")

    # Organiser les entrées utilisateur en colonnes
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=100)
        bmi = st.number_input('BMI', min_value=0.0, max_value=50.0)
        children = st.number_input('Children', min_value=0, max_value=10)

    with col2:
        sex = st.selectbox('Sex', ['male', 'female'])
        smoker = st.selectbox('Smoker', ['yes', 'no'])
        region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    if st.button('Predict'):
        prediction = predict(data)
        st.session_state['predictions'].append({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': prediction[0]
        })
        st.session_state['last_prediction'] = prediction[0]
        set_page("View Data")

# Page View Data
elif st.session_state['page'] == "View Data":
    st.title('View and Download Predictions')
    st.write("### Here you can view your prediction history and download the data.")
    
    if 'last_prediction' in st.session_state:
        st.write('Your predicted insurance charges:')
        st.write(f'{st.session_state["last_prediction"]:.2f}')

    if st.session_state['predictions']:
        df_predictions = pd.DataFrame(st.session_state['predictions'])
        st.dataframe(df_predictions)
        
        # Préparer les données CSV pour le téléchargement
        csv = df_predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
        
        fig = px.line(df_predictions, x=df_predictions.index, y='charges', title='Predicted Insurance Charges Over Time')
        st.plotly_chart(fig)
    else:
        st.write("No predictions yet.")

    # Afficher un bouton pour revenir à la page Welcome
    if st.button('Back to Welcome Page'):
        set_page("Welcome")

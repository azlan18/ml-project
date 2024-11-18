import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load the IMDB dataset
@st.cache_data
def load_data():
    # Read dataset
    data = pd.read_csv("imdb_dataset.csv")
    
    # Renaming relevant columns
    data.rename(columns={'Released_Year':'Release Year',
                         'Certificate':'Age Rating',
                         'IMDB_Rating':'IMDB Rating',
                         'Meta_score':'Metascore',
                         'No_of_Votes':'Votes',
                         'Gross':'Gross Revenue'}, inplace=True)
    
    # Removing rows where Gross Revenue is null
    data = data[data['Gross Revenue'].notnull()]
    
    # Standardizing Age Rating
    data['Age Rating'] = data['Age Rating'].map({
        'U':'U', 'G':'U', 'PG':'U', 'GP':'U', 'TV-PG':'U',
        'UA':'UA', 'PG-13':'UA', 'U/A':'UA', 'Passed':'UA', 'Approved':'UA',
        'A':'A', 'R':'A'
    })
    data = data[data['Age Rating'].notnull()]
    
    # Filtering and cleaning Release Year
    data = data[data['Release Year'].str.match(r'\d\d\d\d')]
    data['Release Year'] = data['Release Year'].astype(int)
    
    # Converting Runtime
    data['Runtime'] = data['Runtime'].str[:-4].astype(int)
    
    # Converting Gross Revenue to millions
    data['Gross Revenue'] = data['Gross Revenue'].str.replace(',','').astype(float) * (10**-6)
    
    # Count and primary genre
    data['Genres'] = data['Genre'].apply(lambda x: len(x.split(', ')))
    data['Primary Genre'] = data['Genre'].str.split(', ').str[0]
    data.drop('Genre', axis=1, inplace=True)
    
    # Binary Metascore existence
    data['Metascore Exists'] = data['Metascore'].notnull()
    data.drop('Metascore', axis=1, inplace=True)
    
    return data

# Create preprocessing transformer
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Age Rating', 'Primary Genre']),
            ('num', 'passthrough', ['Release Year', 'Runtime', 'IMDB Rating', 'Votes', 'Metascore Exists', 'Genres'])
        ]
    )

# Create model pipeline
def create_model(model_type):
    if model_type == 'Linear Regression':
        regressor = LinearRegression()
    else:  # Random Forest
        regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
    
    return Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('regressor', regressor)
    ])

# Load the data
data = load_data()

# Preparing the data
X = data[['Release Year', 'Age Rating', 'Runtime', 'IMDB Rating', 'Votes', 'Metascore Exists', 'Genres', 'Primary Genre']]
y = data['Gross Revenue']

# Streamlit UI
st.title("IMDB Movie Box Office Prediction")
st.image("https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_UX67_CR0,0,67,98_AL_.jpg", width=300)
st.markdown("### Predict a movie's box office collection using its attributes.")

# Model Selection
model_type = st.selectbox("Select Prediction Model", 
                           ['Linear Regression', 'Random Forest'])

# Input sliders and dropdowns
col1, col2 = st.columns(2)

with col1:
    release_year = st.slider("Release Year", 1900, 2024, 2000)
    runtime = st.slider("Runtime (minutes)", 30, 300, 120)
    imdb_rating = st.slider("IMDB Rating", 0.0, 10.0, 7.5, step=0.1)
    genres = st.slider("Number of Genres", 1, 5, 2)

with col2:
    age_rating = st.selectbox("Age Rating", ["U", "UA", "A"])
    votes = st.number_input("Number of Votes (in millions)", 0.1, 10.0, 1.0, step=0.1) * 10**6
    metascore_exists = st.checkbox("Metascore Exists", value=True)
    primary_genre = st.selectbox("Primary Genre", data['Primary Genre'].unique())

# Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the selected model
model = create_model(model_type)
model.fit(X_train, y_train)

# Feature Importance Visualization
if st.checkbox("Show Feature Importance"):
    # Extract feature names after preprocessing
    feature_names = (
        list(model.named_steps['preprocessor'].named_transformers_['cat']
             .get_feature_names_out(['Age Rating', 'Primary Genre']).tolist()) +
        ['Release Year', 'Runtime', 'IMDB Rating', 'Votes', 'Metascore Exists', 'Genres']
    )
    
    # Get feature importances (works for Random Forest, will be uniform for Linear Regression)
    if model_type == 'Random Forest':
        importances = model.named_steps['regressor'].feature_importances_
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importances
        st.bar_chart(importance_df.set_index('feature')['importance'])
    else:
        st.write("Feature importance is not available for Linear Regression")

# Prediction
if st.button("Predict Box Office Collection"):
    input_data = pd.DataFrame({
        'Release Year': [release_year],
        'Age Rating': [age_rating],
        'Runtime': [runtime],
        'IMDB Rating': [imdb_rating],
        'Votes': [votes],
        'Metascore Exists': [metascore_exists],
        'Genres': [genres],
        'Primary Genre': [primary_genre]
    })
    
    prediction = model.predict(input_data)
    
    # Display prediction with formatting
    st.success(f"Predicted Box Office Collection: ${prediction[0]:.2f} Million")

    # Model performance metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    st.info(f"Model Performance ({model_type}) - Train R²: {train_score:.2f}, Test R²: {test_score:.2f}")
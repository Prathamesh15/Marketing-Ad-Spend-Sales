import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define the Streamlit app
@st.cache_data
def predict_sales(tv_budget, radio_budget, newspaper_budget):
    # Create a pandas dataframe with the input variables
    input_df = pd.DataFrame({'TV': [tv_budget],
                             'Radio': [radio_budget],
                             'Newspaper': [newspaper_budget]})
    # Make a prediction with the loaded model
    prediction = model.predict(input_df)
    # Return the prediction as a float
    return float(prediction[0])

# Set page config
st.set_page_config(page_title="Advertising Sales Prediction App", page_icon=":bar_chart:", layout="wide")

# Set app title
st.title("Advertising Sales Prediction App")

# Load an image
image = Image.open("Cost-Effective.jpg")
st.image(image, use_column_width=True)

# Create a sidebar
st.sidebar.header("Enter Budget Values")
tv_budget = st.sidebar.number_input("TV Budget", value=0)
radio_budget = st.sidebar.number_input("Radio Budget", value=0)
newspaper_budget = st.sidebar.number_input("Newspaper Budget", value=0)

# Create a button to predict sales
if st.sidebar.button("Predict Sales"):
    # Call the prediction function with the user inputs
    prediction = predict_sales(tv_budget, radio_budget, newspaper_budget)
    
    # Display the prediction
    st.subheader("Prediction")
    st.success(f"The predicted sales is {prediction:.2f} dollars.")
    
    # Visualize the data
    st.subheader("Data Visualization")
    df = pd.DataFrame({'Media': ['TV', 'Radio', 'Newspaper'], 'Budget': [tv_budget, radio_budget, newspaper_budget]})
    fig = px.bar(df, x='Media', y='Budget', color='Media')
    st.plotly_chart(fig)

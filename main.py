import joblib
import streamlit as st
import pandas as pd
from sklearn.linear_model import LassoCV
import numpy as np
from scipy import stats

#Loading up the Regression model we created
model = joblib.load('model.pkl')


# Define the prediction function
def predict(has_swimmingpools, has_pool, num_rating, num_reviews,
            rating_review_combined, rating_ratio, facilities_count):
  transformed_data = stats.boxcox(np.array([num_rating + 1.1]),
                                  lmbda=1.5019669454621212)

  num_rating = transformed_data[0]
  num_reviews = np.sqrt(num_reviews)
  rating_ratio = np.sqrt(rating_ratio)
  rating_review_combined = np.sqrt(rating_review_combined)

  prediction = model.predict(
    pd.DataFrame([[
      has_swimmingpools, has_pool, num_rating, num_reviews,
      rating_review_combined, rating_ratio, facilities_count
    ]],
                 columns=[
                   'has_swimmingpools', 'has_pool', 'num_rating',
                   'num_reviews', 'rating_review_combined', 'rating_ratio',
                   'facilities_count'
                 ]))
  return prediction


st.title('Hotel Price Predictor')
st.header('Enter the characteristics of the hotel:')
num_rating = st.number_input('Number of Rating:',
                             min_value=0.1,
                             max_value=10.0,
                             value=1.0)
num_reviews = st.number_input('Number of Reviews:', min_value=0.1, value=1.0)

location = st.selectbox('Location:', [
  'Stokol', 'Machlessvile', 'Wanderland', 'Uberlandia', 'Hallerson',
  'Willsmian', 'Andeman', 'Ubisville'
])

st.write('Provided facilities (Can more than one):')
option_1 = st.checkbox('Swimming Pools')
option_2 = st.checkbox('Pools')
option_3 = st.checkbox('Restaurant')
option_4 = st.checkbox('Bar')
option_5 = st.checkbox('Gym')
option_6 = st.checkbox('Internet')
facilities_count = option_1 + option_2 + option_3 + option_4 + option_5 + option_6

has_swimmingpools = 0
has_pool = 0
rating_ratio = np.where((num_rating == 0.0) & (num_reviews == 0), 0.0,
                        num_rating / num_reviews)
rating_review_combined = np.where((num_rating == 0.0) & (num_reviews == 0),
                                  0.0, num_rating * num_reviews)

if (option_1 == 1):
  has_swimmingpools = 1

if (option_2 == 1):
  has_pool = 1

if st.button('Predict Price'):
  price = predict(has_swimmingpools, has_pool, num_rating, num_reviews,
                  rating_review_combined, rating_ratio, facilities_count)

  st.success(
    f'The predicted price of the hotel in {location} is {price[0]:,.2f} avg/night',
    icon="✅")

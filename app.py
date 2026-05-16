import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData


st.title("Student Exam Performance Prediction")

# Input Fields
gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

race_ethnicity = st.selectbox(
    "Race or Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

lunch = st.selectbox(
    "Lunch",
    [
        "free/reduced",
        "standard"
    ]
)

test_preparation_course = st.selectbox(
    "Test Preparation Course",
    [
        "none",
        "completed"
    ]
)

reading_score = st.number_input(
    "Reading Score",
    min_value=0,
    max_value=100
)

writing_score = st.number_input(
    "Writing Score",
    min_value=0,
    max_value=100
)


# Prediction Button
if st.button("Predict Math Score"):

    try:

        data = CustomData(

            gender=gender,

            race_ethnicity=race_ethnicity,

            parental_level_of_education=
            parental_level_of_education,

            lunch=lunch,

            test_preparation_course=
            test_preparation_course,

            reading_score=reading_score,

            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        st.success(
            f"Predicted Math Score: {results[0]:.2f}"
        )

    except Exception as e:
        st.error(f"Error: {e}") 
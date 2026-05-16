import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            # Model Paths
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            print("Before Loading")

            # Load Model and Preprocessor
            model = load_object(file_path=model_path)

            preprocessor = load_object(
                file_path=preprocessor_path
            )

            print("After Loading")

            # Transform Data
            data_scaled = preprocessor.transform(features)

            print("Data Scaled Successfully")

            # Prediction
            preds = model.predict(data_scaled)

            print("Prediction Completed")

            return preds

        except Exception as e:
            print("Prediction Error:", e)
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = (
            parental_level_of_education
        )

        self.lunch = lunch

        self.test_preparation_course = (
            test_preparation_course
        )

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "gender": [self.gender],

                "race_ethnicity": [
                    self.race_ethnicity
                ],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [self.lunch],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [
                    self.reading_score
                ],

                "writing_score": [
                    self.writing_score
                ]
            }

            df = pd.DataFrame(
                custom_data_input_dict
            )

            print(df)

            return df

        except Exception as e:
            print("DataFrame Error:", e)
            raise CustomException(e, sys)
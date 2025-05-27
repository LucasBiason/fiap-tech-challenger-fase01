import os
import cloudpickle
import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, 'wcmodel.pkl')
training_data_path = os.path.join(project_dir, 'data/insurance.csv')

        
class WealthCostPrediction:
    """
    Class responsible for loading, creating and training the insurance cost prediction model.
    """

    def __init__(self):
        """
        Initializes the WealthCostModelCreator with the model and data paths.
        """
        self.pipeline = None
    
    def add_weight_condition(self, X):
        """
        Adds the 'weight_condition' column to the DataFrame based on BMI.
        """
        X["weight_condition"] = ""
        X.loc[X["bmi"] < 18.5, "weight_condition"] = "Underweight"
        X.loc[(X["bmi"] >= 18.5) & (X["bmi"] < 25), "weight_condition"] = "Normal Weight"
        X.loc[(X["bmi"] >= 25) & (X["bmi"] < 30), "weight_condition"] = "Overweight"
        X.loc[X["bmi"] >= 30, "weight_condition"] = "Obese"
        return X

    def create_model(self):
        """
        Creates, trains, and saves the insurance cost prediction model.
        """
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['age', 'bmi', 'children']),
            ('cat', OneHotEncoder(), ['smoker', "weight_condition"])
        ])

        pipeline = Pipeline(steps=[
            ('add_weight_condition', FunctionTransformer(
                lambda x: self.add_weight_condition(x), validate=False
            )),
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=7,
                max_depth=20,
                max_leaf_nodes=17,
                criterion='friedman_mse',
                random_state=1
            ))
        ])

        data = pd.read_csv(training_data_path)
        data = data.drop(['region'], axis='columns')
        data = data.drop(['sex'], axis='columns')

        X = data.drop(['charges'], axis=1)
        y = data['charges']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        pipeline.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            cloudpickle.dump(pipeline, f)

    def predict(self, features):
        """
        Predicts the insurance charge based on the provided features.

        Args:
            features (list or pd.DataFrame): Input features for prediction.

        Returns:
            float: Predicted insurance charge.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please train the model first."
            )
            
        with open(model_path, 'rb') as f:
            self.pipeline = cloudpickle.load(f)
            
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame([features], columns=['smoker', 'age', 'bmi', 'children'])
            
        prediction = self.pipeline.predict(features)
        return round(float(prediction[0]), 2)


if __name__ == "__main__":
    model_creator = WealthCostPrediction()
    model_creator.create_model()
    
    prediction = WealthCostPrediction()
    result = prediction.predict(['no', 33,  25, 0])
    print(f"Predicted insurance cost: {result}")
    
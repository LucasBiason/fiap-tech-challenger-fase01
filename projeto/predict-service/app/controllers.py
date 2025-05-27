from .ml.wealth_cost_prediction import WealthCostPrediction


class PredictController:
    """
    Controller class responsible for handling prediction requests.
    """

    def __init__(self):
        """
        Initializes the PredictController with a WealthCostPrediction instance.
        """
        self.predictor = WealthCostPrediction()

    def predict(self, data):
        """
        Predicts the insurance charge based on the provided data.

        Args:
            data (InsuranceData): Input data for prediction.

        Returns:
            float: Predicted insurance charge.
        """
        features = [data.smoker, data.age, data.bmi, data.children]
        if not all(map(lambda x : x is not None, features)):
            raise ValueError("No features provided for prediction.")
        return self.predictor.predict(features)
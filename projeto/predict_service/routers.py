from fastapi import APIRouter, HTTPException

from .schemas import InsuranceData
from .controllers import PredictController

router = APIRouter()


@router.post("/predict")
def predict(data: InsuranceData):
    """
    Endpoint to predict insurance charge based on input data.

    Args:
        data (InsuranceData): Input data for prediction.

    Returns:
        dict: Predicted charge value.

    Raises:
        HTTPException: If model file is not found or other errors occur.
    """
    try:
        charge = PredictController().predict(data)
        return {"charge": charge}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
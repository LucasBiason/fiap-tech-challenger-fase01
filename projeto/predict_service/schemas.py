from pydantic import BaseModel


class InsuranceData(BaseModel):
    """
    Schema representing the input data for insurance charge prediction.
    """
    smoker: str
    age: int
    bmi: float
    children: int
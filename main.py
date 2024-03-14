from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Trabajo 6",
    description="Fabiola Díaz",
    version="0.0.1"
)

model = joblib.load("model/raisin_logistic_regression_v01.pkl")


@app.post("/api/v1/predict-raisin", tags=["raisin"])
async def predict(
    area: float,
    majoraxislength: float,
    minoraxislength: float,
    eccentricity: float,
    convexarea: float,
    extent: float,
    perimeter: float
):
    dictionary = {
        'Area': area,
        'MajorAxisLength': majoraxislength,
        'MinorAxisLength': minoraxislength,
        'Eccentricity': eccentricity,
        'ConvexArea': convexarea,
        'Extent': extent,
        'Perimeter': perimeter
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        prediction_value = int(prediction[0])
        prediction_value = str(prediction[0])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction_value}
        )

    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )

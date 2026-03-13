from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
import pandas as pd
from io import BytesIO
import os

from src.pipeline.prediction_pipeline import PredictionPipeline

router = APIRouter()

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

pipeline = PredictionPipeline()


@router.post("/predict-ui")
def predict_ui(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: str = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: str = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: str = Form(...),
    TotalCharges: str = Form(...)
):

    try:
        input_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }

        result = pipeline.predict(input_data)

        # Check if request is AJAX
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JSONResponse({
                "prediction": result["prediction"],
                "churn_probability": result["churn_probability"]
            })

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": result["prediction"],
                "probability": result["churn_probability"]
            }
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        print(traceback.format_exc())
        
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JSONResponse(
                {"error": error_msg},
                status_code=400
            )
        
        return JSONResponse(
            {"error": error_msg},
            status_code=400
        )


@router.post("/predict-excel")
async def predict_excel(file: UploadFile = File(...)):
    """
    Batch prediction endpoint for Excel files.
    Accept Excel file with customer data and return predictions.
    
    Excel columns should match form fields:
    gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
    Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    """
    try:
        # Read Excel file
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        
        # Validate that required columns are present
        required_columns = {
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        }
        
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            return JSONResponse(
                {"error": f"Missing required columns: {', '.join(missing_cols)}"},
                status_code=400
            )
        
        # Make predictions for each row
        predictions = []
        for idx, row in df.iterrows():
            try:
                input_data = {col: str(row[col]) for col in required_columns}
                result = pipeline.predict(input_data)
                predictions.append({
                    "row": idx + 1,
                    "prediction": result["prediction"],
                    "churn_probability": round(result["churn_probability"], 4),
                    "churn_risk": "High Risk" if result["prediction"] == 1 else "Safe"
                })
            except Exception as e:
                predictions.append({
                    "row": idx + 1,
                    "error": str(e)
                })
        
        # Add predictions to original dataframe
        predictions_df = pd.DataFrame(predictions)
        result_df = pd.concat([df, predictions_df], axis=1)
        
        # Save to Excel
        output = BytesIO()
        result_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        
        # Return Excel file
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=predictions.xlsx"}
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"Batch prediction error: {error_msg}")
        print(traceback.format_exc())
        
        return JSONResponse(
            {"error": error_msg, "detail": traceback.format_exc()},
            status_code=400
        )
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.prediction_api import router


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

templates = Jinja2Templates(directory="src/pipeline/templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

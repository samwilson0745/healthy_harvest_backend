from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from services import ModelService

app = FastAPI()

UPLOAD_DIRECTORY = "uploads"

# make the directory if it doesn't exist
os.makedirs(UPLOAD_DIRECTORY,exist_ok=True)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...),t:str=Form(...)):
    try:
        # Save the uploaded file
        contents = await file.read()
        #print(contents)

        filename = os.path.join(UPLOAD_DIRECTORY,file.filename)
        with open(filename, "wb") as f:
            f.write(contents)
        model = ModelService(t)
        prediction_result = model.predict_image(filename)
        os.remove(filename)
        return JSONResponse(content=prediction_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import pandas as pd
import io

from predict import predict_batch_csv 

app = FastAPI(title="Coachable Event Predictor")

@app.post("/predict_csv")
async def process_csv_file(file: UploadFile = File(...)):
    contents = await file.read()
    client_df = pd.read_csv(io.BytesIO(contents))
    result_df = predict_batch_csv(client_df, threshold=0.7)
    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    response = Response(content=stream.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=Scored_{file.filename}"
    return response
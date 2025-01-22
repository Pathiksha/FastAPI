from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load your provided model file
model = tf.keras.models.load_model("test_model(1).h5")

def preprocess_image(image: Image.Image):
    """
    Preprocess the image to fit the model's input requirements.
    This includes resizing and normalizing the image.
    """
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0   # Normalize the image if required by the model
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit the model's input shape
    return image

def predict_animal(image: Image.Image):
    """
    Predict whether the image is of an animal or a bird.
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    # Interpret the predictions - assuming a binary classification
    result = "Animal" if predictions[0][0] > 0.5 else "Bird"
    return result

@app.get("/")
def hello():
    return "hello"


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to handle image uploads and return predictions.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_animal(image)
    return JSONResponse(content={"prediction": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


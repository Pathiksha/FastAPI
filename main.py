from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Load your model
model = tf.keras.models.load_model("test_model.h5")

# Define your class labels (use the exact labels from your training data)
class_labels = ["bird", "animal"]  # Update with the correct list of labels

def preprocess_image(image: Image.Image):
    """
    Preprocess the image to fit the model's input requirements.
    """
    image = image.resize((224, 224))  # Adjust size if different for your model
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to handle image uploads and return predictions.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Assume a multi-class prediction; take the index of the highest confidence
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]

    return JSONResponse(content={"prediction": predicted_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

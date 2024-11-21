from fastapi import FastAPI, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import File, UploadFile
model = YOLO("best.pt")
app = FastAPI()

product_prices = {
    "Bru-Coffee": 150,    # Example price
    "Choco-Pie": 45,      # Example price
    "CloseUp": 35,        # Example price
    "Coca-Cola": 50,      # Example price
    "Cocoa Powder": 120,  # Example price
    "Diet-Coke": 60,      # Example price
    "Hersheys": 200,     # Example price
    "Hide-n-Seek": 40,   # Example price
    "KeraGlo": 250,      # Example price
    "Lays": 60,          # Example price
    "Loreal": 400,       # Example price
    "Maggi": 30,         # Example price
    "Marie Light": 50,   # Example price
    "Oreo": 80,          # Example price
    "Pears": 45,         # Example price
    "Pedigree": 150,     # Example price
    "Perk": 35,          # Example price
    "Pringles": 90,      # Example price
    "Yippee": 25,        # Example price
    "Colgate": 60        # Example price
}

def detect_and_get_price(label: str):
    # Check if the label is in the product_prices dictionary
    if label in product_prices:
        return product_prices[label]  # Return the random price
    else:
        return "Product not found"

@app.get("/")
async def read_root():
 return {"message": "Hello, World!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    try:
        # Read the uploaded file
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Uploaded file is not a valid image")
        
        # Perform object detection with YOLOv8
        results = model.predict(image)
        
        # Extract detection details
        detections = []
        for result in results:
            for box in result.boxes:  # Assuming 'boxes' contains detected objects
                label = result.names[int(box.cls)]  # Map class index to label
                confidence = box.conf  # Confidence score
                cost = product_prices[label]  # Bounding box coordinates
                
                detections.append({
                    "label": label,
                    "cost" : detect_and_get_price(label),
                    "confidence": round(float(confidence), 2) * 100 ,
                    
                })
        
        # Return the detections
        return {"detections": detections}

    except Exception as e:
        # Log the error and return a 500 status code
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
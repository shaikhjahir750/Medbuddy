import io
import json
from fastapi.testclient import TestClient
from main import app
from PIL import Image

client = TestClient(app)

def test_predict_image():
    # Create a dummy image
    image = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Test "eye" model
    response = client.post(
        "/predict/image",
        data={"model_type": "eye"},
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    print("Eye Response status:", response.status_code)
    try:
        print("Eye Response body:", response.json())
    except Exception as e:
        print("Eye Response body (raw):", response.text)
        print("Eye JSON decode error:", e)

    # Test "skin" model
    response = client.post(
        "/predict/image",
        data={"model_type": "skin"},
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    print("Skin Response status:", response.status_code)
    try:
        print("Skin Response body:", response.json())
    except Exception as e:
        print("Skin Response body (raw):", response.text)
        print("Skin JSON decode error:", e)

if __name__ == "__main__":
    test_predict_image()

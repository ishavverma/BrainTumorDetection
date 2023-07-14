from flask import Flask, render_template_string, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

app = Flask(__name__)
classification_model = load_model('BrainTumorDetection\Classification\Classification.h5')

def prediction(out):
    if out == 0:
        return 'meningioma'
    elif out == 1:
        return 'glioma'
    elif out == 2:
        return 'pituitary tumor'

index_html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Brain Tumor Classification</title>
  </head>
  <body>
    <h1>Brain Tumor Classification</h1>
    <form method="POST" action="/" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <input type="submit" value="Upload">
    </form>
  </body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Brain Tumor Classification - Result</title>
  </head>
  <body>
    <h1>Brain Tumor Classification - Result</h1>
    <p>{{ result }}</p>
  </body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_path = 'temp_image.jpg'
        file.save(file_path)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3)
        img = image.load_img(file_path)
        plt.imshow(img, interpolation='nearest')
        plt.show()
        predictions = classification_model.predict(img_array)
        indices = predictions.argmax()
        result = prediction(indices)
        return render_template_string(result_html, result=result)
    return render_template_string(index_html)

if __name__ == '__main__':
    app.run()

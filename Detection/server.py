from flask import Flask, render_template_string, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from matplotlib.pyplot import imshow

app = Flask(__name__)
model = load_model('BrainTumorDetection\Detection\Detection.h5')

def names(number):
    if number == 0:
        return 'Tumor Detected'
    else:
        return 'No Tumor Detected'

index_html = """
<!DOCTYPE html>
<html>
  <head>
    <title>Brain Tumor Detection</title>
  </head>
  <body>
    <h1>Brain Tumor Detection</h1>
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
    <title>Brain Tumor Detection - Result</title>
  </head>
  <body>
    <h1>Brain Tumor Detection - Result</h1>
    <img src="{{ result['image_path'] }}" alt="Uploaded Image">
    <p>{{ result['classification'] }}</p>
    <p>{{ result['prediction'] }}% Conclusion</p>
  </body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        x = np.array(img.resize((128, 128)))
        x = x.reshape(1, 128, 128, 3)
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        result = {
            'prediction': str(res[0][classification] * 100),
            'classification': names(classification),
            'image_path': file.filename
        }
        return render_template_string(result_html, result=result)
    return render_template_string(index_html)

if __name__ == '__main__':
    app.run()

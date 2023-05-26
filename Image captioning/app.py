from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os

app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = os.urandom(24)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class ImageForm(FlaskForm):
    image = StringField('Image', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()  # Create an instance of the form
    if request.method == 'POST':
        # Check if file is uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No image file found.')

        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', error='No image file selected.')

        try:
            raw_image = Image.open(image).convert("RGB")
        except Exception as e:
            return render_template('index.html', error='Error opening image file.')

        # Conditional image captioning
        text = "a photography of"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption1 = processor.decode(out[0], skip_special_tokens=True)

        text = "in this photo"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption2 = processor.decode(out[0], skip_special_tokens=True)

        text = "it shows"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption3 = processor.decode(out[0], skip_special_tokens=True)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption4 = processor.decode(out[0], skip_special_tokens=True)
        return render_template('index.html', form=form, caption1=caption1, caption2=caption2,
                               caption3=caption3, caption4=caption4, image_path=image.filename)

    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
model.eval()

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/generate_captions', methods=['POST'])
def generate_captions():
    image_file = request.files['image']
    image_path = 'static/' + image_file.filename
    image_file.save(image_path)
    image = Image.open(image_path)

    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=-1)
    topk_probabilities, topk_indices = torch.topk(probabilities, k=5)


    predicted_class_labels = []
    predicted_class_probabilities = []
    for i in range(3):
        predicted_class_idx = topk_indices[i].item()
        predicted_class_prob = topk_probabilities[i].item()
        predicted_class_label = model.config.id2label[predicted_class_idx]
        predicted_class_labels.append(predicted_class_label)
        predicted_class_probabilities.append(predicted_class_prob)


    captions = []
    for i in range(3):
        caption = f"{predicted_class_labels[i]}: {predicted_class_probabilities[i]:.3f}"
        captions.append(caption)

    return render_template('output.html', image_path=image_path, captions=captions)

if __name__ == '__main__':
    app.run(debug=True)
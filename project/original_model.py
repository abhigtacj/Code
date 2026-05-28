import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile

# Load BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    image = Image.fromarray(input_image).convert('RGB')

    # Generate a caption for the image
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Convert caption to speech using gTTS
    tts = gTTS(text=caption)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    return caption, audio_path

# Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Text(label="Generated Caption"),
        gr.Audio(label="Listen to Caption", type="filepath")
    ],
    title="Image Captioning with TTS",
    description="Upload an image to generate a caption and hear it spoken aloud."
)

iface.launch()
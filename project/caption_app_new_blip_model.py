# caption_app.py
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile
import gradio as gr

reload_path = os.path.abspath("blip-flickr8k")
processor = AutoProcessor.from_pretrained(reload_path)
model = BlipForConditionalGeneration.from_pretrained(reload_path)

def caption_image(input_image: np.ndarray):
    image = Image.fromarray(input_image).convert('RGB')
    inputs = processor(images=image, text="a picture of", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    tts = gTTS(text=caption)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    return caption, audio_path

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
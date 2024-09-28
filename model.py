import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq
from colpali.byaldi import VisionTextProcessor
from PIL import Image

# Load the Qwen2-VL 7B model and processor
processor = VisionTextProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Function to perform OCR using Byaldi + Qwen2-VL 7B
def perform_ocr(image):
    # Open and preprocess the image
    image = Image.open(image)
    inputs = processor(image, return_tensors="pt")
    
    # Generate the text from the image
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

# Function to search for a keyword in the extracted text
def search_text(image, keyword):
    extracted_text = perform_ocr(image)
    
    # Search for the keyword in the extracted text
    if keyword.lower() in extracted_text.lower():
        highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
        return highlighted_text
    else:
        return "Keyword not found!"

        

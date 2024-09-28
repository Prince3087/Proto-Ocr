import gradio as gr
from PIL import Image

# Load Qwen2-VL model and Byaldi processor
from transformers import AutoModelForVision2Seq
from colpali.byaldi import VisionTextProcessor

processor = VisionTextProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# OCR function using Qwen2-VL + Byaldi
def perform_ocr(image):
    image = Image.open(image)
    inputs = processor(image, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Keyword search function
def search_text(image, keyword):
    extracted_text = perform_ocr(image)
    if keyword.lower() in extracted_text.lower():
        highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
        return highlighted_text
    else:
        return "Keyword not found!"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# OCR and Search Application using Qwen2-VL 7B Instruct + Byaldi")
    
    image_input = gr.Image(label="Upload Image", type="filepath")
    keyword_input = gr.Textbox(label="Search Keyword")
    ocr_output = gr.Textbox(label="Extracted Text")
    search_output = gr.Textbox(label="Search Results")
    
    ocr_btn = gr.Button("Extract Text")
    search_btn = gr.Button("Search Text")
    
    ocr_btn.click(perform_ocr, inputs=[image_input], outputs=ocr_output)
    search_btn.click(search_text, inputs=[image_input, keyword_input], outputs=search_output)

demo.launch()

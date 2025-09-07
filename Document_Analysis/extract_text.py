from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img, lang="ara") + "\n"
    except:
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img, lang="ara") + "\n"
    return text

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="ara")

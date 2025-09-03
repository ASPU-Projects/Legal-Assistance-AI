import easyocr
import PyPDF2
import requests
import json

# -----------------------------
# 1. استخراج النص من صورة (OCR)
# -----------------------------
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['ar'])
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

# -----------------------------
# 2. استخراج النص من PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# -----------------------------
# 3. استدعاء Ollama للتلخيص
# -----------------------------
def summarize_with_ollama(text, simplify=False):
    # تعليمات للتلخيص أو التبسيط
    if simplify:
        prompt = f"بسط النص التالي ليكون مفهوماً للإنسان العادي:\n\n{text}"
    else:
        prompt = f"لخص النص القانوني التالي باحترافية:\n\n{text}"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "command-r7b-arabic",
            "prompt": prompt
        }
    )

    result_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                result_text += data["response"]
    return result_text.strip()

# -----------------------------
# 4. التشغيل
# -----------------------------
if __name__ == "__main__":
    # اختر صورة أو PDF
    text = extract_text_from_image("photo_1_2025-07-17_11-35-13.jpg")
    # text = extract_text_from_pdf("legal_file.pdf")

    print("📌 النص الأصلي:\n", text[:1000], "...")

    # تلخيص قانوني
    summary = summarize_with_ollama(text, simplify=False)
    print("\n📌 الملخص القانوني:\n", summary)

    # تبسيط بلغة إنسانية
    simple_summary = summarize_with_ollama(summary, simplify=True)
    print("\n📌 التبسيط بلغة إنسانية:\n", simple_summary)

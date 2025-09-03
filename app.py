from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Legal Assistance AI",
              description="""
A Three AI Systems work for this platform 
""")

@app.get("/")
def ready():
    return "Done"

@app.get("/chat_AI")
def Chat(message):
    return message

# @app.get("/get_recommanded/{user_text}")
# def run_get_recommanded(user_text):
#     return recommendation(user_text)

# # Explicitly allow OPTIONS request (optional)
# @app.options("/get_recommanded/{user_text}")
# def options_handler(user_text):
#     return JSONResponse(content=recommendation(user_text), status_code=200)

@app.get("/document_analysis")
def getRecommendation(docuemnt):
    return docuemnt




from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.param_functions import Form
from fastapi.staticfiles import StaticFiles

from starlette.responses import FileResponse 

from hfpipelines import IdiomaticIvan


# Define a dictionary to store the model
# Using a global variable is acceptable here as it is loaded once at startup
# and accessed by all requests.
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and tokenizer at startup
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # Specify device if using GPU (e.g., "cuda:0")
    models["ivan"] = IdiomaticIvan()
    print("ML models loaded successfully!")
    yield
    # Clean up resources on shutdown (optional, but good practice)
    models.clear()
    print("ML models unloaded.")


app = FastAPI(lifespan=lifespan)


@app.api_route("/", methods=["GET", "POST"])
async def index(request: Request):
    if request.method == "POST":
        data = await request.form()
        data = dict(data)
        prompt = data.get("prompt")
        response = models["ivan"].prompt(prompt)
        return {"response": response}

    elif request.method == "GET":
        return FileResponse(
            "static/index.html",
            media_type="text/html",
            status_code=200,
            headers={"Cache-Control": "no-cache"},
            content_disposition_type="inline"
        )


app.mount("/static", StaticFiles(directory="static"), name="static")
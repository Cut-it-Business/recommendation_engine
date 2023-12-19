from typing import List

import uvicorn
from fastapi import File, UploadFile, FastAPI, Depends
from stages.model import Recommended
import asyncio
from functools import partial
from PIL import Image
import io


app = FastAPI()

class AppContext:
    def __init__(self):
        self.model = Recommended('model/best.pt')

context = AppContext()


def blocking_operation(partners):
    if type(partners) == list:
        return context.model.batch_recommended(partners)
    d = context.model.recommended(partners)
    return d

async def run_blocking_operation(partners):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(blocking_operation, partners))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        contents = Image.open(io.BytesIO(image_data))
        response = await run_blocking_operation(contents)
    except Exception as e:
        return {"Exception": e}
    finally:
        file.file.close()

    return response


@app.post('/batch_upload')
async def batch_upload(files: List[UploadFile] = File(...)):
    try:
        images_data = [await file.read() for file in files]
        contents = [Image.open(io.BytesIO(image_data)) for image_data in images_data]
        response = await run_blocking_operation(contents)
    except Exception as e:
        return {"Exception": e}
    finally:
        [file.file.close() for file in files]

    return response

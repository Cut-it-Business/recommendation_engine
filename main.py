import os
import random

from typing import List, Literal

import uvicorn
from fastapi import File, UploadFile, FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles

from stages.model import Recommended
import asyncio
from functools import partial
from PIL import Image
import io


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class AppContext:
    def __init__(self):
        self.model = Recommended('model/clip_emb_logreg.pickle', 'model/sorted_classes.pkl')


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


@app.get('/images')
async def images(request: Request, gender: Literal['male', 'female'] = None):    
    def random_img(gender_type):
        """
        Returns a random image, chosen among the files of the given path.
        """
        files = os.listdir('static/' + gender_type)
        index = random.randrange(0, len(files))
        return files[index]
        
    image_links = set()
    while len(image_links) < 9:
        if gender is None:
            random_gender = random.choice(['male', 'female'])
            img = random_img(random_gender)
            img_url = request.url_for('static', path=f'{random_gender}/{img}')
            image_links.add(str(img_url))
            continue
        img = random_img(gender)
        img_url = request.url_for('static', path=f'{gender}/{img}')
        image_links.add(str(img_url))

    return image_links

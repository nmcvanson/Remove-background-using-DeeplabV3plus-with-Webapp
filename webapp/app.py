import logging
import uuid
import os
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import main
from typing import List
from PIL import Image
import cv2

app = FastAPI()
UPLOAD_FOLDER = "images-input"
OUTPUT_FOLDER = "images-output"
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images-input", StaticFiles(directory=UPLOAD_FOLDER), name="images-input")
app.mount("/images-output", StaticFiles(directory=OUTPUT_FOLDER), name="images-output")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post("/remove-bg", response_class=HTMLResponse)
async def remove_bg(request: Request, file: UploadFile = File(...)):
    try:
        new_name = str(uuid.uuid4()).split("-")[0]
        ext = file.filename.split(".")[-1]
        file_name = f"{UPLOAD_FOLDER}/{new_name}.{ext}"

        with open(file_name, "wb+") as f:
            f.write(file.file.read())

        input_image = Image.open(file_name)
        new_image, masked_image = main.remove_bg_mult(file_name)

        #output_image = os.path.join(OUTPUT_FOLDER, f'{new_name}.png')
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{new_name}.png'), new_image)
        output_image = f"{OUTPUT_FOLDER}/{new_name}.png"

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{new_name}_mask.png'), masked_image)
        #mask_image = f"{OUTPUT_FOLDER}/{new_name}_mask.png"
        return templates.TemplateResponse("results.html", context={"request": request,
                                                                   'original_path': file_name,
                                                                   'filepath': output_image,
                                                                   'img_no_bk': f'{new_name}.png',
                                                                   'mask_img': f'{new_name}_mask.png'})
                                                    
    except Exception as ex:
        logging.info(ex)
        print(ex)
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.post("/change-bg")
async def change_bg(img_no_bk: str = Form(...), mask_img: str = Form(...), file: UploadFile = File(...)):
    try:
        new_name = str(uuid.uuid4()).split("-")[0]
        ext = file.filename.split(".")[-1]
        file_name = f"{UPLOAD_FOLDER}/{new_name}_bk.{ext}"

        with open(file_name, "wb+") as f:
            f.write(file.file.read())

        input_img = Image.open(os.path.join(OUTPUT_FOLDER, img_no_bk)).convert("RGBA")
        input_bk = Image.open(file_name).convert("RGBA")
        mask_image = Image.open(os.path.join(OUTPUT_FOLDER, mask_img)).convert("RGBA")
        new_img = main.change_background(input_img, input_bk, mask_image)

        output_image = os.path.join(OUTPUT_FOLDER, f'{new_name}_with_bk.png')
        new_img.save(output_image)
        
        return JSONResponse(
            status_code=200,
            content={
                'img_with_bk': output_image,
                'img_no_bk': img_no_bk,
            },
        )

    except Exception as ex:
        logging.info(ex)
        print(ex)
        return JSONResponse(status_code=400, content={"error": str(ex)})

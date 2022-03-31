import os
import cv2
import numpy as np
import io
from rembg.bg import remove
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision
from pymatting import cutout
from random import randrange
import threading

from Docker.MODNet.src.models.modnet import MODNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_for_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_for_detection.eval()

diversity = 1000000


def process_request_by_input_output_path(input_path, output_path):
    print("Process img : " + input_path)
    try:
        if contains_people(input_path, model_for_detection):
            use_modnet(input_path, output_path)
            pass
        else:
            if contains_white_bg(input_path):
                use_cv_and_rembg(input_path, output_path)
            else:
                use_rembg(input_path, output_path)
    except Exception:
        result_str = 'Something went wrong'


def clean(path):
    th = threading.Thread(target=os.remove, args=(path,))
    th.start()


def use_cv_and_rembg(input_path, output_path):
    output_path_cv2 = get_path("output_cv2")
    output_path_rembg = get_path("output_rembg")

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    mask = 255 - mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    cv2.imwrite(output_path_cv2, result)

    f = np.fromfile(input_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(output_path_rembg)

    img1 = Image.open(output_path_cv2)
    img2 = Image.open(output_path_rembg)
    img1.paste(img2, (0, 0), mask=img2)
    img1.save(output_path)

    clean(output_path_rembg)
    clean(output_path_cv2)


def use_rembg(input_path, output_path):
    f = np.fromfile(input_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(output_path)


def use_modnet(input_path, output_path):
    device = torch.device("cpu")
    ckpt_path = 'modnet_photographic_portrait_matting.ckpt'
    ref_size = 512
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).to(device)
    modnet.load_state_dict(torch.load(ckpt_path, map_location=device))
    modnet.eval()
    im = Image.open(input_path)
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]
    im = Image.fromarray(im)
    im = im_transform(im)
    im = im[None, :, :, :]
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    _, _, matte = modnet(im.to(device), True)
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte_path = str(randrange(diversity)) + ".png"
    Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(matte_path)
    cutout(input_path, matte_path, output_path)
    clean(matte_path)


def contains_people(path, model):
    print("Check contains people img : " + path)
    img = Image.open(path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    labels = list(pred[0]['labels'].numpy())
    scores = list(pred[0]['scores'].detach().numpy())
    for i in range(len(labels)):
        if labels[i] == 1 and scores[i] > 0.95:
            return True
    return False


def contains_white_bg(path):
    image = Image.open(path)
    pix = image.load()
    a = pix[0, 0][0]
    b = pix[0, 0][1]
    c = pix[0, 0][2]
    if a + b + c > 750:
        return True
    return False


def get_path(tag):
    return str(randrange(diversity)) + "_" + tag + ".png"

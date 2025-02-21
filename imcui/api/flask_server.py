from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

from . import ImagesInput, to_base64_nparray
from .core import ImageMatchingAPI
from ..hloc import DEVICE
from ..ui import get_version


# This is just a simple Flask server that serves the ImageMatchingAPI.
# IMPORTANT: It is not meant to be used in production, but rather as a simple way to test the API.
#
# To launch the server, run the following command in the terminal at the root of the project:
#    python -m imcui.api.flask_server.py
#
# This will start the server on http://localhost:8001.
#
# You can then use the client script to send requests to the server.
#    python -m imcui.api.imw_client.py
#
# If launched from a docker container, run it with the following command:
#     docker run -it -p 7860:7860 -p 8001:8001 vincentqin/image-matching-webui:latest python app.py --server-name "0.0.0.0" --server-port=7860 && python -m imcui.api.flask_server 

app = Flask(__name__)

conf = yaml.safe_load(open(Path(__file__).parent / "config/api.yaml"))
api = ImageMatchingAPI(conf=conf["api"], device=DEVICE)

@app.route("/")
def root():
    return "Hello, world!"

@app.route("/version", methods=["GET"])
def version():
    return jsonify({"version": get_version()})

@app.route("/v1/match", methods=["POST"])
def match():
    try:
        image0 = request.files['image0']
        image1 = request.files['image1']
        image0_array = load_image(image0)
        image1_array = load_image(image1)
        output = api(image0_array, image1_array)
        skip_keys = ["image0_orig", "image1_orig"]
        pred = postprocess(output, skip_keys)
        return jsonify(pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/extract", methods=["POST"])
def extract():
    try:
        input_info = ImagesInput(**request.json)
        preds = []
        for i, input_image in enumerate(input_info.data):
            image_array = to_base64_nparray(input_image)
            output = api.extract(
                image_array,
                max_keypoints=input_info.max_keypoints[i],
                binarize=input_info.binarize,
            )
            skip_keys = []
            pred = postprocess(output, skip_keys)
            preds.append(pred)
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_image(file) -> np.ndarray:
    with Image.open(file) as img:
        return np.array(img)

def postprocess(output: dict, skip_keys: list) -> dict:
    pred = {}
    for key, value in output.items():
        if key in skip_keys:
            continue
        if isinstance(value, np.ndarray):
            pred[key] = value.tolist()
    return pred

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)

from flask import Flask, request, jsonify
import requests
import torch
from PIL import Image
from torchvision import models, transforms

# Load model from models/vgg16.pt
model = models.vgg16(pretrained=False)

# Change the last layer to output 100 classes
model.classifier[-1] = torch.nn.Linear(4096, 100)

model.load_state_dict(
    torch.load("../models/vgg16.pt", map_location=torch.device("cpu"))
)

dict_classes = {
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "crab",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm",
}

app = Flask(__name__)


def get_image_from_url(url: str) -> Image:
    """Get an image from a url and return a tensor"""
    try:
        # Download the image
        image = Image.open(requests.get(url, stream=True).raw)
        return image
    except Exception as e:
        print(e)
        return None


# Endpoint to receive a image and return a prediction
# The client must send a POST request with a JSON body
# {
#   "image": "base64 encoded image"
#   "image_url": "url to image"
# }
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    image_url = request.json["image_url"]

    # Validate if almost one of the parameters is present
    # if image is None and image_url is None:
    if image_url is None:
        return jsonify({"error": "image or image_url must be present"}), 400

    # Get the image from file or url
    # to_pred = image if image is not None else get_image_from_url(image_url)
    to_pred = get_image_from_url(image_url)
    
    if not to_pred:
        return jsonify({"error": "error with image"}), 400

    # Create transform to resize to 32x32 and normalize
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    # Transform the image
    to_pred = transform(to_pred)

    # Add a dimension to the tensor
    to_pred = to_pred.unsqueeze(0)

    # Get the prediction
    pred = model(to_pred)

    # Return class and confidence
    return (
        jsonify(
            {
                "class": pred.argmax().item(),
                "class_name": dict_classes[pred.argmax().item()],
                "confidence": pred.max().item(),
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Run the app locally
    app.run(host="localhost", port=5000, debug=True)

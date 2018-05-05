import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import datasets
from train import RESULT_MODEL_PATH, TEST_PATH, TEST_TRANSFORMS, BODY_NAME, get_body, get_head


def main(image_path):
    image = Image.open(image_path)
    body = get_body(BODY_NAME)
    head = get_head(BODY_NAME)
    body.eval()
    head.eval()
    head.load_state_dict(torch.load(RESULT_MODEL_PATH))
    image_tensor = TEST_TRANSFORMS(image).unsqueeze(0)
    image_var = Variable(image_tensor, volatile=True)
    predicted_class = head(body(image_var)).data.numpy().argmax()
    classes = datasets.ImageFolder(TEST_PATH, TEST_TRANSFORMS).classes
    print("Predict: ", classes[predicted_class])


if __name__ == "__main__":
    FLOWER = os.path.join(TEST_PATH, "flower/0d5b30e623554b4a11df1bce025b4608e0c94e85.jpg")
    RED_PANDA = os.path.join(TEST_PATH, "red_panda/1c636ad23eeb371d78aa48c49ae9fcde546eab5c.jpg")
    BIG_PANDA = os.path.join(TEST_PATH, "big_panda/1c89597750e6f01bcfb986fae221a23e29f33456.jpg")
    FALAFEL = os.path.join(TEST_PATH, "falafel/0e21fd392758637553bbab631da40c22cd376950.jpg")
    main(FLOWER)
    main(RED_PANDA)
    main(BIG_PANDA)
    main(FALAFEL)

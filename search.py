import os
from PIL import Image
from torch.autograd import Variable
from train import TEST_PATH, TEST_TRANSFORMS, BODY_NAME, get_body


def get_nc(body, image_path):
    image = Image.open(image_path)
    image_tensor = TEST_TRANSFORMS(image).unsqueeze(0)
    image_var = Variable(image_tensor, volatile=True)
    nc = body(image_var).data.squeeze().numpy()
    return nc


def main():
    body = get_body(BODY_NAME)
    body.eval()
    RED_PANDA = os.path.join(TEST_PATH, "red_panda/1c636ad23eeb371d78aa48c49ae9fcde546eab5c.jpg")
    print(get_nc(body, RED_PANDA))
    # TODO implement me !!!


if __name__ == "__main__":
    main()

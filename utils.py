import cv2
import torch

def resize_image(image, height, square=False):
    h, w = image.shape[:2]
    if square:
        new_size = max(h, w)
        top = (new_size - h) // 2
        bottom = new_size - h - top
        left = (new_size - w) // 2
        right = new_size - w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        image = cv2.resize(image, (height, height))
    else:
        new_width = int(w * (height / h))
        image = cv2.resize(image, (new_width, height))
    return image

def load_and_resize_image(img, height, square):
    # img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    img = resize_image(img, height, square)
    return img

def process_image(image, model, transform, device):
    img_input = transform({"image": image})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    return prediction

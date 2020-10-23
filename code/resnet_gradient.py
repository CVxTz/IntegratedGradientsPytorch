import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from imagenet_mapping import mapping
from torchvision import transforms
from tqdm import tqdm


def get_model(model_name='resnet18'):
    model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    if torch.cuda.is_available():
        model.to('cuda')

    model.eval()
    return model


def get_transformations():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    return preprocess, resize


def predict_sample(batch, model):
    with torch.no_grad():
        output_x = model(batch)
        v, idx = torch.max(output_x[0], dim=-1)
        idx = idx.cpu().data.numpy()

    return v, idx, mapping[int(idx)]


def move_to_device(batch):
    if torch.cuda.is_available():
        batch = batch.to('cuda')
    return batch


def compute_integrated_gradient(batch_x, batch_blank, model, idx):
    mean_grad = 0
    n = 100

    for i in tqdm(range(1, n + 1)):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)[0, idx]
        grad, = torch.autograd.grad(y, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients


def plot_images(images, titles, output_path, n=2):
    fig, axs = plt.subplots(1, n)

    fig.set_figheight(10)
    fig.set_figwidth(16)

    for i, (title, img) in enumerate(zip(titles, images)):
        axs[i].imshow(img)
        axs[i].set_title(title)
        axs[i].axis('off')

    fig.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    filename_x = "../input/cat_1.jpg"
    filename_blank = "../input/blank.png"

    idx = 281  # tabby cat class

    model = get_model()

    # Read images
    input_image_x = Image.open(filename_x).convert('RGB')
    input_image_blank = Image.open(filename_blank).convert('RGB')

    preprocess, resize = get_transformations()

    # Preprocess inputs
    input_tensor_x = preprocess(input_image_x)
    batch_x = input_tensor_x.unsqueeze(0)

    input_tensor_blank = preprocess(input_image_blank)
    batch_blank = input_tensor_blank.unsqueeze(0)

    resized_tensor_x = resize(input_image_x)
    resized_batch_x = resized_tensor_x.unsqueeze(0)

    batch_x = move_to_device(batch_x)
    batch_blank = move_to_device(batch_blank)

    # Integrated gradient computation

    integrated_gradients = compute_integrated_gradient(batch_x, batch_blank, model, idx)

    # Change to channel last
    integrated_gradients = integrated_gradients.permute(0, 2, 3, 1)
    batch_x = batch_x.permute(0, 2, 3, 1)
    resized_batch_x = resized_batch_x.permute(0, 2, 3, 1)

    # Squeeze + move to cpû

    np_integrated_gradients = integrated_gradients[0, :, :, :].cpu().data.numpy()

    resized_x = resized_batch_x[0, :, :, :].cpu().data.numpy()

    np_integrated_gradients = np.fabs(np_integrated_gradients)

    # normalize amplitudes

    np_integrated_gradients = np_integrated_gradients / np.max(np_integrated_gradients)

    # Overlay integrated gradient with image

    images = [resized_x, np_integrated_gradients]
    titles = ['Tabby Cat', 'Integrated Gradient']

    plot_images(images, titles, "../output/" + filename_x.split("/")[-1])

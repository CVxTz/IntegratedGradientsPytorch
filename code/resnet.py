import torch
from PIL import Image
from torchvision import transforms
from code.imagenet import mapping
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

filename_x = "../input/cat_2.jpg"
filename_blank = "../input/blank.png"

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

model.eval()


url_blank, filename_blank = ("https://www.pixelstalk.net/wp-content/uploads/2016/10/Blank-Wallpaper-Full-HD.png",
                             "blank.png")


# urllib.request.urlretrieve(url_x, filename_x)
# urllib.request.urlretrieve(url_blank, filename_blank)


input_image_x = Image.open(filename_x)
input_image_blank = Image.open(filename_blank).convert('RGB')

print(input_image_blank)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor_x = preprocess(input_image_x)
input_batch_x = input_tensor_x.unsqueeze(0)

input_tensor_blank = preprocess(input_image_blank)
input_batch_blank = input_tensor_blank.unsqueeze(0)

if torch.cuda.is_available():
    input_batch_x = input_batch_x.to('cuda')
    input_batch_blank = input_batch_blank.to('cuda')
    model.to('cuda')


with torch.no_grad():
    output_x = model(input_batch_x)
    output_blank = model(input_batch_blank)
    v, idx = torch.max(output_x[0], dim=-1)
    idx = idx.cpu().data.numpy()
    print(v, idx, mapping[int(idx)])

n = 100
mean_grad = 0
idx = 281  # tabby cat class

for i in tqdm(range(1, n + 1)):
    x = input_batch_blank + i/n * (input_batch_x - input_batch_blank)
    x.requires_grad = True
    y = model(x)[0, idx]
    grad, = torch.autograd.grad(y, x)

    mean_grad += grad / n

integrated_gradients = (input_batch_x - input_batch_blank) * mean_grad

integrated_gradients = integrated_gradients.permute(0, 2, 3, 1)
input_batch_x = input_batch_x.permute(0, 2, 3, 1)

np_integrated_gradients = integrated_gradients[0, :, :, :].cpu().data.numpy()
np_x = input_batch_x[0, :, :, :].cpu().data.numpy()

ksize = (3, 3)
np_integrated_gradients = np.fabs(np_integrated_gradients)
np_integrated_gradients = cv2.blur(np_integrated_gradients, ksize)

np_integrated_gradients = np_integrated_gradients / np.sum(np_integrated_gradients) * np_integrated_gradients.size


plt.figure()
plt.imshow(np_integrated_gradients)
plt.savefig("integrated_gradients.png")

overlay = np_integrated_gradients*np_x
overlay = (overlay - overlay.min())/(overlay.max() - overlay.min())

plt.figure()
plt.imshow(overlay)
plt.savefig("integrated_gradients_overlay.png")
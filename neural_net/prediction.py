from PIL import Image
import torch
from torchvision import transforms
from neural_net.model_setup import model


# prediction function

def make_prediction(model, image, classes):

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.CenterCrop(124),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    ])

    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(image), dim=1).item()

    return classes[pred]


# Load model
model.load_state_dict(torch.load('model.pth'))

# # Load classes for predictions
classes = torch.load('classes.pth')

# # Load image
#img = Image.open('/Users/szokirov/Documents/Datasets/Intel/seg_test/sea/20099.jpg')


#print(make_prediction(model, img, classes))

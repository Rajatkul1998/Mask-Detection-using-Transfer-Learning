from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F
import streamlit as st 
from pretrained_model import vgg16model
loader = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
from PIL import Image


checkpoint = torch.load('saved_model.pt')
vgg16model.load_state_dict(checkpoint['model_state_dict'])


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
  
    image_tensor = loader(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    pred=vgg16model(image_tensor)
    prob=F.softmax(pred,dim=1)
    
    if(prob[0][0]>0.50):
        st.write(" No Mask detected")
    else:    
        st.write("Mask Detected")

    st.image(image)    







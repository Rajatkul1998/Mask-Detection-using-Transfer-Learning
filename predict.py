from torchvision import datasets, models, transforms
import torch
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from streamlit_webrtc import ClientSettings
import torch.nn.functional as F
import streamlit as st 
from pretrained_model import resnet
transform= transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
from PIL import Image

WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

checkpoint = torch.load('resnet18_final_model.pt',map_location=torch.device("cpu"))
resnet.load_state_dict(checkpoint['model_state_dict'])

resnet.eval()
class VideoTransformer(VideoTransformerBase):
    
        def __init__(self):
            self.canvas = None
            self.x1, self.y1 = 0, 0

        def transform(self, frame):
            frame= frame.to_ndarray(format="bgr24")
            #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(frame)
            image=transform(image)
            image=image.unsqueeze(0)
            output=resnet(image)
            _,prediction=torch.max(output,dim=1)
            print(prediction)
            if(prediction.item()==1):
                cv2.putText(frame,"No Mask Detected", (40,40), 2, 2, 255)
            else: 
                cv2.putText(frame,"Mask Detected", (40,40), 2, 2, 255)

            return frame 
def webcam():
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer,client_settings=WEBRTC_CLIENT_SETTINGS,)  

def img_upload():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    
        image_tensor = transform(image).float()
        image_tensor = image_tensor.unsqueeze(0)
        output=resnet(image_tensor)
        #st.write(output)
        _,prediction=torch.max(output,dim=1)
        if(prediction.item()==1):
            st.write("No Mask detected")
                
        else: 
            st.write("Mask Detected")
               

        st.image(image)    

if __name__=="__main__":
    webcam()





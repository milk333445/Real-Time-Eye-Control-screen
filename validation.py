import cv2
import torch
from torchvision import transforms
from models.mobilenet_v1 import MobileNetV1
from models.mobilenet_v2 import MobileNetV2
from models.mobilenet_v3 import MobileNetV3

class Predictor:
    def __init__(self, model_path, resize=(448, 448), device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetV3(num_classes=1)
        try:
            self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
            print(f"Model loaded from {model_path}")
        except:
            print("Model loading failed")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
        ])
    
    def predict(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.resize)
        frame = self.transform(frame)
        frame = frame.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(frame)
        return pred.cpu().numpy()[0][0]
            
if __name__ == "__main__":
    predictor = Predictor("./v3_checkpoints/2024_02_21_13_14_56/best_model_0.001.pth")
    img = cv2.imread("./eye_images/x0.134_y0.447.jpg")
    pred = predictor.predict(img)
    print(pred)
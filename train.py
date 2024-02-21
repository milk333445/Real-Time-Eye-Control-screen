import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.mobilenet_v1 import MobileNetV1
from models.mobilenet_v2 import MobileNetV2
from models.mobilenet_v3 import MobileNetV3
from eye_dataloader import create_eye_dataloader
from torchvision import transforms
import os
import argparse
import time

class Trainer:
    def __init__(self, model, train_loader, test_loader, save_folder="checkpoints", criterion=None, optimizer=None, lrs=[0.001], device=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_folder = save_folder
        self.criterion = criterion if criterion else nn.MSELoss()
        self.lrs = lrs
        self.device = device
        self.now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) # ex: 2021_07_20_14_30_00
        self.model.to(self.device)
        
        
    def train(self, num_epoches):
        best_loss = float("inf")
        all_train_losses = []
        all_test_losses = []
        
        for lr in self.lrs:
            # initialize weights
            try:
                self.model._initialize_weights()
            except:
                print("Model does not have _initialize_weights method")
                break   
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            best_loss = float("inf")
            train_losses = []
            test_losses = []
        
            for epoch in range(num_epoches):
                self.model.train()
                running_loss = 0.0
                
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                
                epoch_loss = running_loss / len(self.train_loader)
                train_losses.append(epoch_loss)
                test_loss = self.evaluate()
                test_losses.append(test_loss)
                
                print(f"LR: {lr}, Epoch: {epoch+1}/{num_epoches}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_path = f"best_model_{lr}.pth"
                    self.save_model(save_path, epoch, best_loss)
            
            all_train_losses.append(train_losses)
            all_test_losses.append(test_losses)
            
        self.plot_losses(all_train_losses, all_test_losses, self.lrs)
        
    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)
        
    def save_model(self, path, epoch, loss):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(f"{self.save_folder}/{self.now_time}"):
            os.makedirs(f"{self.save_folder}/{self.now_time}")
        path = f"{self.save_folder}/{self.now_time}/{path}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")
        
    def plot_losses(self, all_train_losses, all_test_losses, lrs):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for i, losses in enumerate(all_train_losses):
            plt.plot(losses, label=f"LR: {lrs[i]}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, losses in enumerate(all_test_losses):
            plt.plot(losses, label=f"LR: {lrs[i]}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Testing Loss')
        plt.legend()

        plt.tight_layout()
        
        plot_save_path = f"{self.save_folder}/{self.now_time}/losses_plot.png"
        plt.savefig(plot_save_path)
        print(f"Losses plot saved to {plot_save_path}")
        
        plt.show()
        
                
def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
            
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    if args.model == "v1":
        model = MobileNetV1(num_classes=args.num_classes).to(device)
    elif args.model == "v2":
        model = MobileNetV2(num_classes=args.num_classes).to(device)
    elif args.model == "v3":
        model = MobileNetV3(num_classes=args.num_classes).to(device)
    else:
        raise ValueError("Model version must be v1, v2, or v3")
    
    
    train_loader, test_loader = create_eye_dataloader(args.data_path, args.predict, args.batch_size, args.img_size, args.test_split)
    save_folder = f"{args.model}_checkpoints"
    trainer = Trainer(model, train_loader, test_loader, save_folder, device=device, lrs=args.lrs)
    trainer.train(args.num_epoches)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetVx model")
    parser.add_argument("--model", type=str, default="v3", help="Model version, v1, v2, or v3")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes in the dataset")
    parser.add_argument("--data_path", type=str, default="eye_images", help="Path to the train images folder")
    parser.add_argument("--predict", type=str, default="x", help="Predict x or y")
    parser.add_argument("--img_size", type=int, default=448, help="Image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split")
    parser.add_argument("--num_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.001], help="Learning rate")
    args = parser.parse_args()
    main(args)
    
        
                

import os
import io
import base64
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import cv2

from cifar100_labels import CIFAR100_CLASSES, get_display_name, get_superclass_name

class ResNet50CIFAR100Predictor:
    def __init__(self, model_path: str = 'resnet50_cifar100_finetuned.pth', device_str: str = None):
        """
        Loads fine-tuned ResNet-50 model trained on CIFAR-100.
        Auto-detects CUDA / MPS / CPU device.
        """
        self.model_path = model_path
        
        if device_str:
            self.device = torch.device(device_str)
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
                
        print(f"[ModelLoader] Initializing ResNet-50 CIFAR-100 model on device: {self.device}")
        
        # Build ResNet-50 architecture with 100-class output head
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(2048, 100)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
            
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Standard training transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        
    def preprocess_bgr_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocesses OpenCV BGR frame (numpy array):
        1. Resize to (256, 256)
        2. Convert BGR to RGB
        3. Convert to Tensor & Normalize
        4. Add batch dimension
        """
        # Resize to 256x256
        resized = cv2.resize(frame_bgr, (256, 256))
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Transform to Tensor & normalize
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    def preprocess_pil_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocesses PIL Image object."""
        pil_rgb = pil_img.convert('RGB').resize((256, 256))
        tensor = self.transform(pil_rgb).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, frame_bgr: np.ndarray, top_k: int = 5):
        """
        Runs model inference on an OpenCV BGR frame.
        Returns:
            dict containing top_prediction and list of top_k results.
        """
        tensor = self.preprocess_bgr_frame(frame_bgr)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, top_k)
            
        results = []
        for prob, idx in zip(top_probs, top_indices):
            c_idx = idx.item()
            c_name = CIFAR100_CLASSES[c_idx]
            conf = float(prob.item() * 100)
            results.append({
                "class_index": c_idx,
                "class_name": c_name,
                "display_name": get_display_name(c_idx),
                "superclass": get_superclass_name(c_name),
                "confidence": round(conf, 2)
            })
            
        top_pred = results[0]
        return {
            "top_prediction": top_pred,
            "top_k": results,
            "device": str(self.device)
        }

    def predict_base64(self, base64_str: str, top_k: int = 5):
        """Processes base64 encoded image string (e.g. from Web frontend canvas/webcam)."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        pil_img = Image.open(io.BytesIO(img_bytes))
        # Convert PIL to BGR numpy array
        rgb_np = np.array(pil_img.convert('RGB'))
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        return self.predict(bgr_np, top_k=top_k)

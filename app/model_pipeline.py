import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import pickle

# --- Model Definitions (Copied from notebook to ensure pickle compatibility) ---
class SRCNN(nn.Module):
    def __init__(self, activation="relu", residual=False):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.PReLU()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            act,
            nn.Conv2d(64, 32, 5, padding=2),
            act,
            nn.Conv2d(32, 1, 5, padding=2)
        )

    def forward(self, x):
        out = self.net(x)
        return x + out if self.residual else out

class ImprovedSRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.act2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.act3 = nn.PReLU()
        self.conv4 = nn.Conv2d(16, 1, kernel_size=5, padding=2)
        self.res_scale = 0.1

    def forward(self, x):
        identity = x
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.conv4(out)
        return identity + self.res_scale * out

# --- Pipeline Class ---

class ModelPipeline:
    def __init__(self, model_path=None):
        self.model = None
        self.device = 'cpu' # Force CPU for this demo, or check cuda if needed
        # Default path relative to project root (where run.py is executed)
        default_path = os.path.join('SRCNN', 'models', 'base_srcnn_model.pkl')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists(default_path):
            print(f"Found default model at {default_path}")
            self.load_model(default_path)
        else:
            print("Warning: No model found. Pipeline will fail if called.")

    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        try:
            # --- Pickle Namespace Fix ---
            # The model was likely saved in a notebook where SRCNN was defined in __main__.
            # We need to map __main__.SRCNN to our local SRCNN class.
            import sys
            import __main__
            
            # Save original __main__ attributes if any (optional safety)
            # Inject our classes into __main__
            setattr(__main__, 'SRCNN', SRCNN)
            setattr(__main__, 'ImprovedSRCNN', ImprovedSRCNN)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model_map = {
                'SRCNN': SRCNN,
                'ImprovedSRCNN': ImprovedSRCNN,
            }

            state = model_data.get('model_state_dict') or model_data.get('state_dict')
            if state is None:
                raise KeyError(f"No 'model_state_dict' found in {model_path}")

            # Auto-detect model type based on keys
            keys = list(state.keys())
            if any('net' in k for k in keys):
                print("Detected SRCNN architecture keys.")
                self.model = SRCNN()
            else:
                print("Defaulting to ImprovedSRCNN architecture.")
                self.model = ImprovedSRCNN()

            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def process_image(self, input_path, output_path, scale_factor=4):
        if not self.model:
            print("Model not loaded, cannot process.")
            return False

        try:
            import time
            start_time = time.time()
            print(f"Starting processing for {input_path}")

            # STRICT implementation matching SRCNN/inference.ipynb enhance_image
            
            # 1. Read image
            img = cv2.imread(str(input_path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {input_path}")

            # 2. Color conversion
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y_channel = img_ycrcb[:, :, 0]

            h, w = y_channel.shape
            print(f"Image size: {w}x{h}")
            
            # 3. Resize Y channel (Bicubic) for input
            y_bicubic = cv2.resize(y_channel, (w * scale_factor, h * scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
            
            # 4. Prepare tensor
            y_tensor = torch.tensor(y_bicubic / 255.).float().unsqueeze(0).unsqueeze(0)
            y_tensor = y_tensor.to(self.device)

            # 5. Inference
            t0 = time.time()
            with torch.no_grad():
                sr_y = self.model(y_tensor).cpu().squeeze().numpy()
            print(f"Inference time: {time.time() - t0:.2f}s")

            # 6. Post-process Y channel
            sr_y = np.clip(sr_y * 255., 0, 255).astype(np.uint8)

            # 7. Resize Cr and Cb channels (Bicubic)
            cr_channel = cv2.resize(img_ycrcb[:, :, 1], (w * scale_factor, h * scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)
            cb_channel = cv2.resize(img_ycrcb[:, :, 2], (w * scale_factor, h * scale_factor), 
                                   interpolation=cv2.INTER_CUBIC)

            # 8. Merge channels
            sr_ycrcb = np.stack([sr_y, cr_channel, cb_channel], axis=2)
            sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

            # 9. Save result
            cv2.imwrite(output_path, sr_bgr)
            
            print(f"Total processing time: {time.time() - start_time:.2f}s")
            return True

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return False

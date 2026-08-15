import os
import time
import cv2
import numpy as np

from model_loader import ResNet50CIFAR100Predictor

class DesktopCVApp:
    def __init__(self, camera_id: int = 0, model_path: str = 'resnet50_cifar100_finetuned.pth'):
        self.camera_id = camera_id
        self.predictor = ResNet50CIFAR100Predictor(model_path=model_path)
        self.realtime_mode = False
        self.last_prediction = None
        self.snapshot_dir = 'snapshots'
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
    def draw_hud(self, frame: np.ndarray, prediction_data: dict, fps: float) -> np.ndarray:
        """
        Draws professional Computer Vision HUD overlay on top of frame.
        """
        h, w, c = frame.shape
        hud = frame.copy()
        
        # 1. Semi-transparent header bar
        header_height = 80
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        
        # Header title & device
        cv2.putText(hud, "LIVE VISION AI", (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (56, 189, 248), 2)
        device_str = f"Device: {self.predictor.device.type.upper()} | FPS: {fps:.1f}"
        cv2.putText(hud, device_str, (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        
        # Mode badge (Continuous vs Snapshot)
        mode_text = "MODE: REAL-TIME (Press 'r' to toggle)" if self.realtime_mode else "MODE: MANUAL CAPTURE (Press 'c' to predict)"
        mode_color = (34, 197, 94) if self.realtime_mode else (251, 146, 60)
        cv2.putText(hud, mode_text, (w - 360, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

        # 2. Draw Target Bounding Box in Center
        box_size = int(min(h, w) * 0.55)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Target box corners
        corner_len = 25
        thick = 3
        color_corner = (56, 189, 248)
        
        # Corners top-left
        cv2.line(hud, (x1, y1), (x1 + corner_len, y1), color_corner, thick)
        cv2.line(hud, (x1, y1), (x1, y1 + corner_len), color_corner, thick)
        # Corners top-right
        cv2.line(hud, (x2, y1), (x2 - corner_len, y1), color_corner, thick)
        cv2.line(hud, (x2, y1), (x2, y1 + corner_len), color_corner, thick)
        # Corners bottom-left
        cv2.line(hud, (x1, y2), (x1 + corner_len, y2), color_corner, thick)
        cv2.line(hud, (x1, y2), (x1, y2 - corner_len), color_corner, thick)
        # Corners bottom-right
        cv2.line(hud, (x2, y2), (x2 - corner_len, y2), color_corner, thick)
        cv2.line(hud, (x2, y2), (x2, y2 - corner_len), color_corner, thick)
        
        cv2.putText(hud, "TARGET REGION", (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (148, 163, 184), 1)

        # 3. Draw Prediction Results if available
        if prediction_data:
            top_pred = prediction_data['top_prediction']
            conf = top_pred['confidence']
            disp_name = top_pred['display_name']
            super_cls = top_pred['superclass']
            
            # Confidence banner background
            banner_y = header_height + 15
            banner_h = 60
            
            banner_bg = (16, 185, 129) if conf >= 40 else ((245, 158, 11) if conf >= 15 else (100, 116, 139))
            overlay_b = hud.copy()
            cv2.rectangle(overlay_b, (20, banner_y), (360, banner_y + banner_h), (30, 41, 59), -1)
            cv2.addWeighted(overlay_b, 0.85, hud, 0.15, 0, hud)
            cv2.rectangle(hud, (20, banner_y), (25, banner_y + banner_h), banner_bg, -1)
            
            # Text info
            cv2.putText(hud, f"PREDICTION: {disp_name.upper()}", (35, banner_y + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(hud, f"Confidence: {conf:.1f}% | {super_cls}", (35, banner_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)
            
            # 4. Top-5 Bar Chart (Right side overlay panel)
            panel_w = 260
            panel_x = w - panel_w - 20
            panel_y = header_height + 15
            panel_h = 220
            
            overlay_p = hud.copy()
            cv2.rectangle(overlay_p, (panel_x, panel_y), (w - 20, panel_y + panel_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay_p, 0.85, hud, 0.15, 0, hud)
            
            cv2.putText(hud, "TOP-5 PROBABILITIES", (panel_x + 15, panel_y + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (56, 189, 248), 1)
            
            bar_start_y = panel_y + 40
            for idx, item in enumerate(prediction_data['top_k']):
                cur_y = bar_start_y + idx * 34
                name = item['display_name']
                if len(name) > 14:
                    name = name[:12] + ".."
                c_val = item['confidence']
                
                # Class name & percentage
                cv2.putText(hud, f"{name}", (panel_x + 15, cur_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (226, 232, 240), 1)
                cv2.putText(hud, f"{c_val:.1f}%", (panel_x + panel_w - 60, cur_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)
                
                # Bar graphic
                bar_max_w = panel_w - 30
                bar_fill_w = int((c_val / 100.0) * bar_max_w)
                bar_fill_w = max(bar_fill_w, 2)
                
                bar_color = (56, 189, 248) if idx == 0 else (100, 116, 139)
                cv2.rectangle(hud, (panel_x + 15, cur_y + 16), (panel_x + 15 + bar_max_w, cur_y + 20), (51, 65, 85), -1)
                cv2.rectangle(hud, (panel_x + 15, cur_y + 16), (panel_x + 15 + bar_fill_w, cur_y + 20), bar_color, -1)

        # 5. Footer Instructions Bar
        cv2.rectangle(hud, (0, h - 35), (w, h), (15, 23, 42), -1)
        instructions = "[C] Predict Frame | [R] Toggle Real-time | [S] Save Snapshot | [Q] Quit"
        cv2.putText(hud, instructions, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1)
                    
        return hud

    def run(self):
        """Starts webcam loop and key handler."""
        print(f"[DesktopCV] Opening webcam device {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam index {self.camera_id}.")
            print("Please ensure your camera is connected and permissions are granted.")
            return

        print("\n==================================================")
        print("  LIVE COMPUTER VISION PREDICTOR STARTED")
        print("--------------------------------------------------")
        print("  Press 'c' to capture & predict current frame")
        print("  Press 'r' to toggle real-time continuous mode")
        print("  Press 's' to save snapshot to snapshots/")
        print("  Press 'q' to quit application")
        print("==================================================\n")

        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            prev_time = curr_time

            # Crop central bounding box for inference
            h, w, _ = frame.shape
            box_size = int(min(h, w) * 0.55)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            crop_target = frame[y1:y1+box_size, x1:x1+box_size]

            # Real-time mode inference
            if self.realtime_mode:
                self.last_prediction = self.predictor.predict(crop_target)

            # Render HUD
            hud_frame = self.draw_hud(frame, self.last_prediction, fps)
            cv2.imshow("ResNet-50 CIFAR-100 Live AI Computer Vision", hud_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[DesktopCV] Quitting application...")
                break
            elif key == ord('c'):
                print("\n[DesktopCV] Capturing & processing current frame...")
                self.last_prediction = self.predictor.predict(crop_target)
                top = self.last_prediction['top_prediction']
                print(f" -> Top Prediction: {top['display_name']} ({top['confidence']}%) [{top['superclass']}]")
            elif key == ord('r'):
                self.realtime_mode = not self.realtime_mode
                status = "ENABLED" if self.realtime_mode else "DISABLED"
                print(f"[DesktopCV] Real-time inference mode {status}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%HM%S")
                filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.png")
                cv2.imwrite(filename, hud_frame)
                print(f"[DesktopCV] Saved snapshot image to: {filename}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DesktopCVApp()
    app.run()

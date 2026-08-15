import argparse
import os
import cv2

from model_loader import ResNet50CIFAR100Predictor

def predict_single_image(image_path: str, top_k: int = 5):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    predictor = ResNet50CIFAR100Predictor()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file '{image_path}'.")
        return

    result = predictor.predict(frame, top_k=top_k)
    
    print("\n==================================================")
    print(f"  IMAGE PREDICTION RESULTS: {os.path.basename(image_path)}")
    print(f"  Hardware Device: {result['device'].upper()}")
    print("--------------------------------------------------")
    top = result['top_prediction']
    print(f"  TOP PREDICTION: {top['display_name'].upper()} ({top['confidence']}%)")
    print(f"  Category:       {top['superclass']}")
    print("--------------------------------------------------")
    print("  TOP-5 PROBABILITIES:")
    for idx, item in enumerate(result['top_k'], 1):
        print(f"   {idx}. {item['display_name']:<18} | {item['confidence']:>6.2f}% | [{item['superclass']}]")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CIFAR-100 class for a single image file.")
    parser.add_argument("image_path", help="Path to input image file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display (default: 5)")
    args = parser.parse_args()
    
    predict_single_image(args.image_path, top_k=args.top_k)

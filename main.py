import argparse
import sys
import uvicorn

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-50 CIFAR-100 Live Computer Vision Project Launcher"
    )
    parser.add_argument(
        "--mode",
        choices=["desktop", "web"],
        default="desktop",
        help="Run mode: 'desktop' for OpenCV webcam GUI window, or 'web' for web app server (default: desktop)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for Web server mode (default: 8000)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam camera index for desktop mode (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50_cifar100_finetuned.pth",
        help="Path to trained PyTorch model state dict file"
    )
    
    args = parser.parse_args()

    if args.mode == "desktop":
        print(f"\n[Launcher] Starting Desktop OpenCV Computer Vision Application...")
        from desktop_cv import DesktopCVApp
        app = DesktopCVApp(camera_id=args.camera, model_path=args.model)
        app.run()

    elif args.mode == "web":
        print(f"\n[Launcher] Starting FastAPI Web Computer Vision Studio on http://127.0.0.1:{args.port}...")
        import web_app
        uvicorn.run("web_app:app", host="127.0.0.1", port=args.port, reload=False)

if __name__ == "__main__":
    main()

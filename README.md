# AQUAVISION

AQUAVISION is an underwater AI enhancement and detection system built with FastAPI. It processes uploaded videos and images using a TensorRT UNet enhancement model and a YOLO detection model for marine object and naval mine detection.

## Features

- Web-based upload flow for video and image files
- Underwater enhancement using a TensorRT UNet engine
- Object detection and tracking with YOLO
- Live frame streaming via Server-Sent Events (SSE)
- Optional HUD overlay and mine warning display
- Processed outputs saved to `outputs/`
- Built-in health, status, progress, and download endpoints

## Project structure

- `main.py` - FastAPI application, page routes, upload endpoints, processing workflow, and SSE streaming
- `inference_module.py` - Core model loading and processing logic
- `requirements.txt` - Python dependencies
- `index.html`, `upload.html`, `results.html`, `about.html` - Static frontend pages
- `uploads/` - Uploaded input files
- `outputs/` - Generated processed files
- `unet_fp16.engine` - TensorRT UNet engine used for underwater enhancement
- `marine_yolo26m_v1/weights/last.engine` - YOLO TensorRT engine for object detection

## Requirements

- Python 3.10
- CUDA-capable GPU with matching CUDA/cuDNN support for PyTorch and TensorRT
- TensorRT 10.16.0.72
- PyTorch with CUDA support

Install Python dependencies from the provided file:

```bash
python -m pip install -r requirements.txt
```

> Note: `torch` and `torchvision` are configured for CUDA 13.0 in `requirements.txt`. Install the matching CUDA-enabled wheel for your environment.

## Running the app


run with Uvicorn directly:

```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

Then open the app in your browser:

```text
http://127.0.0.1:8080/
```

## Usage

- `GET /` - Home page
- `GET /upload` - Upload page
- `GET /results` - Results page
- `POST /api/upload` - Upload a video or image file
- `POST /api/process` - Start video processing
- `POST /api/process-image` - Process a single image
- `GET /api/stream/{job_id}` - Stream processed video frames via SSE
- `GET /api/progress/{job_id}` - Poll processing progress
- `GET /api/download/{filename}` - Download processed output
- `GET /api/list-outputs` - List processed files
- `GET /health` - Service health check
- `GET /api/status` - Service status summary
- `POST /api/warmup` - Warm up models in the background


## Weights

Download the required model weights and place them in the specified directories:

### U-Net Weights
- [Download U-Net Weights](https://drive.google.com/file/d/1MF-iSZY1xkioKjvLUqkTcHxrHWzGNZL5/view?usp=drivesdk)

### YOLO Weights
- [Download YOLO Weights](https://drive.google.com/drive/folders/1FsbNLREkbM8DutZoWhFFmaElH6z_tG-7)

## Notes

- Uploaded files are saved in `uploads/`.
- Processed outputs are saved in `outputs/`.
- For best performance, use an NVIDIA GPU supported by TensorRT.

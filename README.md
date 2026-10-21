# Face Detection ML

A Python webcam face-detection app with a browser-based interface, designed for running locally from VS Code. It uses Streamlit for the interface, Streamlit-WebRTC for webcam frames, and OpenCV's pre-trained Haar Cascade face detector.

## Features

- Real-time webcam stream
- Multiple face detection
- Bounding boxes and per-frame face count
- Start/stop camera controls
- Dark interface and detector notes
- No identity recognition and no intentional frame storage

## Run in VS Code on Windows

Use Python 3.10 or 3.11.

1. Extract the project ZIP.
2. Open the extracted `FaceDetectionML` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Run these commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt terminal instead:

```bat
.venv\Scripts\activate.bat
```

Streamlit should open a browser tab at a local address, usually `http://localhost:8501`. Click **START** and allow camera access.

## Troubleshooting

- **Camera permission:** allow camera access in Windows privacy settings and in the browser.
- **WebRTC does not start:** try Chrome or Edge and refresh the local app page.
- **Port is busy:** run `streamlit run app.py --server.port 8502`.
- **Installation issue:** use Python 3.10 or 3.11 and reinstall from `requirements.txt`.

## How it works

OpenCV's Haar Cascade model detects frontal-face patterns in each webcam frame. It is a classic, fast baseline for learning computer vision. It is not a modern neural-network detector and can miss faces at angles, in low light, or at small sizes.

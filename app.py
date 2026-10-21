import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})
st.set_page_config(
    page_title="Face Detection ML",
    page_icon="👁️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e8eef7; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .hero {
        padding: 1.2rem 1.4rem; border-radius: 18px;
        background: linear-gradient(120deg, #14243a, #12352f);
        border: 1px solid #263b50; margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; color: #f2f7ff; }
    .hero p { color: #b8c8d8; margin: .35rem 0 0 0; }
    .small-note { color: #9caec1; font-size: .9rem; }
</style>
<div class="hero">
  <h1>👁️ Face Detection ML</h1>
  <p>Real-time multi-face detection using Python and OpenCV.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(model_path)
    if detector.empty():
        raise RuntimeError("OpenCV face-detection model could not be loaded.")
    return detector

detector = load_detector()

class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_count = 0
        self.enabled = True
        self.detector = load_detector()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(35, 35),
        )
        self.face_count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (80, 245, 160), 2)
            cv2.putText(
                image, "FACE", (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 245, 160), 2,
                cv2.LINE_AA,
            )

        cv2.rectangle(image, (12, 12), (245, 52), (18, 31, 47), -1)
        cv2.putText(
            image, f"Faces detected: {self.face_count}", (22, 39),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 245, 255), 2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

with st.sidebar:
    st.header("Settings")
    st.markdown("**Detector:** OpenCV Haar Cascade")
    st.markdown("**Mode:** Live webcam")
    st.markdown("**Processing:** Local Python process")
    st.divider()
    st.caption("Haar Cascade is lightweight and useful for learning. It may miss side profiles or faces in poor lighting.")

left, right = st.columns([1.6, 1])
with left:
    st.subheader("Live camera")
    st.write("Allow camera access when your browser asks.")
      webrtc_ctx = webrtc_streamer(
        key="face-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )
    st.caption("Press START to begin and STOP to end the camera stream.")

with right:
    st.subheader("How it works")
    st.markdown("""
    1. The browser provides webcam frames.
    2. OpenCV converts each frame to grayscale.
    3. The detector locates faces.
    4. Green rectangles are drawn around detections.
    """)
    st.info("This project detects face locations. It does not identify people.")
    st.subheader("Privacy")
    st.write("The app does not save frames or intentionally upload them to a separate server. In local use, frames are processed by the Python app running on your computer.")

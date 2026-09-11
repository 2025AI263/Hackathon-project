"""
app.py
------
A small Streamlit app to TEST Person 2's face detection + recognition
module on its own (Person 4 will later fold this logic into the full
team dashboard alongside liveness detection).

CAMERA NOTE (read this):
Streamlit's built-in st.camera_input() takes ONE SNAPSHOT per click —
it is not a continuous live video stream like a desktop OpenCV window
(cv2.imshow). This is a deliberate, practical choice for a hackathon:
  - It works out-of-the-box on Streamlit Community Cloud (uses the
    browser's camera, no server-side webcam access needed).
  - True continuous real-time video in Streamlit requires the extra
    `streamlit-webrtc` package and more setup — a good "stretch goal"
    if you have time, but not required for a working demo.
For the demo: click the camera button, take a photo, see the result.

Run locally with:
    streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st

from face_recognition.recognizer import FaceRecognizer

st.set_page_config(page_title="Face Recognition Module - Person 2", layout="centered")


@st.cache_resource
def load_recognizer():
    # @st.cache_resource makes sure the (somewhat heavy) ML model is
    # loaded from disk ONLY ONCE, and reused across every Streamlit
    # rerun (Streamlit reruns this whole script on every button click /
    # interaction). Without this, the app would reload the model every
    # single time you click anything, which is slow and wasteful.
    return FaceRecognizer()


recognizer = load_recognizer()


def uploaded_file_to_cv2_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


st.title("Face Detection & Recognition — Person 2 Module")
st.caption("Standalone test dashboard. Not the final team app.")

page = st.sidebar.radio("Page", ["Register a user", "Recognize a face", "Registered users"])

# ---------------- REGISTER PAGE ----------------
if page == "Register a user":
    st.header("Register a new user")
    name = st.text_input("Name")
    st.write("Take 3–5 photos, moving your head slightly between shots "
             "(straight, slight left, slight right) for a more robust match.")
    num_samples = st.slider("Number of photos to take", min_value=1, max_value=5, value=3)

    captured_images = []
    for i in range(num_samples):
        photo = st.camera_input(f"Sample {i + 1}", key=f"enroll_sample_{i}")
        if photo is not None:
            captured_images.append(uploaded_file_to_cv2_image(photo))

    if st.button("Register", type="primary"):
        if not name.strip():
            st.error("Please enter a name first.")
        elif len(captured_images) == 0:
            st.error("Please take at least one photo before registering.")
        else:
            with st.spinner("Generating face embeddings..."):
                result = recognizer.enroll(name.strip(), captured_images)

            if result["success"]:
                st.success(f"{result['message']} — {result['samples_saved']} face sample(s) saved.")
                if result["samples_rejected"] > 0:
                    st.warning(f"{result['samples_rejected']} photo(s) were rejected "
                               f"(no face, or more than one face detected).")
            else:
                st.error(result["message"])

# ---------------- RECOGNIZE PAGE ----------------
elif page == "Recognize a face":
    st.header("Recognize a face")
    photo = st.camera_input("Take a photo")

    if photo is not None:
        frame = uploaded_file_to_cv2_image(photo)
        result = recognizer.recognize(frame)

        if not result.get("face_detected"):
            st.warning(result.get("message", "No face detected."))
        elif result.get("face_count", 0) > 1:
            st.warning(result.get("message"))
        else:
            x1, y1, x2, y2 = result["bbox"]
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected face")

            if result.get("message") == "No registered users yet":
                st.info("No registered users yet — go register someone first.")
            elif result["recognized"]:
                st.success(f"AUTHORIZED — {result['name']}")
            else:
                st.error("UNKNOWN person")

            if result.get("distance") is not None:
                st.write(f"Recognition distance: `{result['distance']:.3f}` "
                         f"(threshold: `{recognizer.threshold}`)")
                st.write(f"Confidence: `{result.get('confidence', 0.0)}%`")

# ---------------- USERS PAGE ----------------
elif page == "Registered users":
    st.header("Registered users")
    users = recognizer.db.list_users()
    if users:
        for u in users:
            st.write(f"- **{u}** ({recognizer.db.count_samples(u)} sample(s))")
    else:
        st.info("No one is registered yet.")

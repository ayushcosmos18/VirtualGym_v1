import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp
import concurrent.futures

app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
executor = concurrent.futures.ThreadPoolExecutor()


@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})


@socketio.on("image")
def receive_image(image):
    # Decode the base64-encoded image data
    image = base64_to_image(image)

    # Process the received image with pose estimation asynchronously
    socketio.start_background_task(process_and_emit, image)


def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def process_and_emit(image):
    # Process the image with pose estimation
    processed_image = process_pose_estimation(image)

    # Emit the processed image
    socketio.emit("processed_image", processed_image)


def process_pose_estimation(image):
    results = pose_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Encode the processed image to base64
    _, buffer = cv2.imencode('.jpg', image)
    processed_img_data = base64.b64encode(buffer).decode()

    return "data:image/jpg;base64," + processed_img_data


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8000, host='0.0.0.0')

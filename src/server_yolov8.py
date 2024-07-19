from flask import Flask, render_template, Response, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
from ultralytics import YOLO
import numpy as np
import asyncio
import threading

# Create a Flask app instance
app = Flask(__name__, static_url_path='/static')

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")  # Ensure this is the correct path to your YOLOv8 model

# Set to keep track of RTCPeerConnection instances
pcs = set()

# Thread-safe frame buffer
frame_buffer = []
frame_lock = threading.Lock()

# Function to generate video frames from the camera with YOLO detection
def generate_frames():
    camera = cv2.VideoCapture(2)  # Replace with your video feed URL
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        # Update the frame buffer
        with frame_lock:
            frame_buffer.append(frame)
        
        # Remove old frames to prevent buffer overflow
        if len(frame_buffer) > 10:
            frame_buffer.pop(0)

        # Yield the latest frame
        with frame_lock:
            if frame_buffer:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer[-1] + b'\r\n')

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Asynchronous function to handle offer exchange
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Create and set the local description
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

# Wrapper function for running the asynchronous offer function
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
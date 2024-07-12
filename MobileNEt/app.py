from flask import Flask, request, jsonify, send_file, render_template
from moviepy.editor import VideoFileClip, ImageSequenceClip
import tensorflow as tf
import pathlib
import os
import mimetypes
import cv2
import time  # Import time module

app = Flask(__name__)

# Function to load the TensorFlow model
def load_model(model_name):
    current_dir = pathlib.Path().resolve()
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True,
        cache_subdir=current_dir)
    model_dir = pathlib.Path(model_dir) / "saved_model"
    return model_dir

# Function to perform object detection
def detect_objects(frame, detection_function):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis,...]
    detections = detection_function(input_tensor)
    return detections

# Function to draw bounding boxes and labels
def draw_boxes(frame, detections, confidence_threshold=0.5):
    height, width, _ = frame.shape

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    for i in range(num_detections):
        score = detections['detection_scores'][i]
        class_id = int(detections['detection_classes'][i])
        bbox = detections['detection_boxes'][i]

        if score < confidence_threshold or class_id != 1:
            continue

        ymin, xmin, ymax, xmax = bbox
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return frame

# Load TensorFlow model on startup
model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
model_dir = load_model(model_name)
detection_model = tf.saved_model.load(str(model_dir))
detection_function = detection_model.signatures['serving_default']

# Route to handle video upload and processing
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_person():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check if the file type is supported
    mime_type, _ = mimetypes.guess_type(file.filename)
    if not mime_type or mime_type.split('/')[0] != 'video':
        return jsonify({'error': 'Unsupported file type'})

    # Choose a directory where you want to save the uploaded video temporarily
    video_directory = 'outputs'  # Update this path as per your environment
    os.makedirs(video_directory, exist_ok=True)  # Ensure the directory exists

    # Save uploaded video to a temporary location
    video_path = os.path.join(video_directory, 'uploaded_video.mp4')
    file.save(video_path)
    
    # Record time when the file is received
    time_received = time.time()

    # Check file size for debugging
    print(f"Uploaded file size: {os.path.getsize(video_path)} bytes")

    # Open the video file using MoviePy
    clip = VideoFileClip(video_path)

    # Process each frame and perform object detection
    processed_frames = []
    frame_count = 0
    for frame in clip.iter_frames():
        frame_count += 1
        # Process every 5th frame
        if frame_count % 5 != 0:
            continue
        
        # Perform object detection on the frame
        detections = detect_objects(frame, detection_function)
        
        # Draw bounding boxes and labels on the frame (only for persons)
        frame_with_boxes = draw_boxes(frame.copy(), detections)

        processed_frames.append(frame_with_boxes)

    # Create a MoviePy ImageSequenceClip from the processed frames
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    
    # Write the processed video to an output file
    output_path = os.path.join(video_directory, 'output_video.mp4')
    processed_clip.write_videofile(output_path, codec='libx264')  # Adjust codec as needed

    # Close the original clip and processed clip
    clip.close()
    processed_clip.close()

    # Record time when processing completes and calculate response time
    time_processed = time.time()
    response_time = time_processed - time_received

    # Check size of output video for debugging
    print(f"Output video size: {os.path.getsize(output_path)} bytes")

    # Rename output file to 'processed_video.mp4'
    os.rename(output_path, os.path.join(video_directory, 'output_video.mp4'))
    output_path = os.path.join(video_directory, 'output_video.mp4')

    # Return the processed video file and response time to the user
    print(f"Model's Response Time: {response_time}")
    return send_file(output_path, as_attachment=True, mimetype='video/mp4')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

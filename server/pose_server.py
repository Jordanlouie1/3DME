import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
# Use relative paths in current working directory
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pose(image_path):
    """Process an image and detect pose landmarks"""
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)

        # Draw pose landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract landmark data
            landmarks_data = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'id': idx,
                    'name': mp_pose.PoseLandmark(idx).name,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            return image, landmarks_data, True
        else:
            return image, [], False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        processed_image, landmarks, pose_detected = process_pose(filepath)

        # Save processed image
        processed_filename = f'processed_{filename}'
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, processed_image)

        # Convert processed image to base64 for display
        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'pose_detected': pose_detected,
            'landmarks_count': len(landmarks),
            'landmarks': landmarks,
            'processed_image': img_base64,
            'filename': processed_filename
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/processed/<filename>')
def get_processed(filename):
    """Serve processed images"""
    return send_file(
        os.path.join(app.config['PROCESSED_FOLDER'], filename),
        mimetype='image/jpeg'
    )

@app.route('/list')
def list_files():
    """List all uploaded and processed files"""
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    processed = os.listdir(app.config['PROCESSED_FOLDER'])
    return jsonify({
        'uploads': uploads,
        'processed': processed
    })

if __name__ == '__main__':
    print("Starting Pose Processing Server...")
    print(f"Upload folder: ./{UPLOAD_FOLDER}")
    print(f"Processed folder: ./{PROCESSED_FOLDER}")
    app.run(host='0.0.0.0', port=5200, debug=True)
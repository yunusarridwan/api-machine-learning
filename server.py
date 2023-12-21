import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

from model_1_files.inference import ImageInference

UPLOAD_FOLDER = 'model_1_files/upload_video'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/user/<id>/predict', methods=['POST'])
def upload_video(id):
    # check file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video = request.files['file']

    # check file name not null
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # check file extension for video is allowed
    if not allowed_file(video.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # use secure file name
    filename = secure_filename(video.filename)
    
    # path to save video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # save video
    video.save(video_path)

    # Process video ImageInference
    inference_instance = ImageInference(video_filename=filename)
    curl_counter = inference_instance.get_curl_counter()
    curl_counter_wrong = inference_instance.get_curl_counter_wrong()

    # remove video file after processing
    os.remove(video_path)

    result = {
        'curl_counter': curl_counter,
        'curl_counter_wrong': curl_counter_wrong
    }

    return jsonify({'message': 'Video berhasil diunggah!', 'result': result}), 201

if __name__ == '__main__':
    app.run(port=8080,debug=True)
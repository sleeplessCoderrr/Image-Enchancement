import os
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, current_app
from .utils import save_uploaded_file
from .model_pipeline import ModelPipeline

main = Blueprint('main', __name__)

pipeline = ModelPipeline() 

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        filename = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'])
        if filename:
            input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            output_filename = 'processed_' + filename
            output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)
            
            success = pipeline.process_image(input_path, output_path)
            
            if success:
                return redirect(url_for('main.result', filename=output_filename, original=filename))
            
    return render_template('index.html')

@main.route('/result')
def result():
    processed_file = request.args.get('filename')
    original_file = request.args.get('original')
    return render_template('result.html', processed=processed_file, original=original_file)

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

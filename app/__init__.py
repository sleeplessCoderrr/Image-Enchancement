import os
from flask import Flask

def create_app():
    app = Flask(__name__)
    
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)
    
    return app

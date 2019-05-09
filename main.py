from flask import Flask, make_response, request
from flask_cors import CORS
import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask import render_template
import subprocess

UPLOAD_FOLDER = './inputs/'

app = Flask(__name__, static_url_path='/inputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/app')
def form():
    return """
        <html>
            <body>
                <h1>Submit an image with closed eye.</h1>
                <form action="/result" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" />
                    <input type="submit" />
                </form>
            </body>
        </html>
    """

@app.route('/result', methods=["POST"])
def transform_view():
    request_file = request.files['data_file']
    if not request_file:
        return "No file"
    # if os.path.splitext(request_file.filename)[1] != '.png':
    #     return "Please reupload an png image!"
    request_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.png'))
    
    # do mask detection processing.
    # like `python ./model/run.py --input ./inputs/test.png --output ./masks/mask.png`
    # input: ./inputs/test.png
    # output: ./masks/mask.png
    preprocess_script = ["python", "./model/mask.py", "--input", "./inputs/test.png", "--output", "./masks/mask.png"]
    output = subprocess.run(preprocess_script)
    # output = subprocess.check_output([preprocess_script], shell=True)

    # do model generation. 
    # `python ./model/test.py --checkpoints ./model/checkpoints/celeba --input ./inputs/test.png --mask ./masks/mask.png --output ./results --model=3`
    query = ["python", "./model/test.py", "--checkpoints", "./model/checkpoints/celeba", "--input", "./inputs/test.png", "--mask", "./masks/mask.png", "--output" , "./results", "--model=3"]
    output = subprocess.run(query)
    print(output)
    # output = subprocess.check_output([query], shell=True)

    return send_from_directory('./results/', 'test.png') # after finish processing, route to /result and serve result image.

if __name__ == '__main__':
    # app.run(debug=True, host = '127.0.0.1', port = 5000)
    from gevent.pywsgi import WSGIServer
    app.debug = True 
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
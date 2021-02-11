#!/usr/bin/python3

'''face-identifier.py runs a simple web server as a means of
interacting with the camera on a Raspberry Pi and the OpenCV facial
recognition models.  This is part of the "face recognition
exploration" described in this notebook:

https://colab.research.google.com/drive/1HtlU8EZYyz7YvhddN_rk7emAFmE2H9ig?authuser=1#scrollTo=wcoqCe9dkI8w

Copyright 2021, John R. Frank jrf.ttst@gmail.com

'''


import http.server
import io
import os
import operator
import re
import time
import traceback
import yaml

import cv2
from collections import defaultdict, Counter
from hashlib import sha256
import matplotlib as mpl
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import matplotlib.patches as patches
import numpy as np
import picamera
import picamera.array


this_dir = '/home/pi/faces'  #os.getcwd()
model_metadata_file_path = os.path.join(this_dir, 'model_metadata.yml')
face_model_path = os.path.join(this_dir, 'face_identification_model.yml')
current_image_name = 'current_image.jpg'
current_image_file_path = os.path.join(this_dir, current_image_name)

class Models(object):
    'Models encapsulates the loading of model data files.'
    
    def __init__(self):
        self.clear()

    def clear(self):
        self._face_detector = None
        self._face_recognizer = None
        self._ids_to_names = None
        self._num_training_examples = None

    @property
    def face_detector(self):
        if self._face_detector is None:
            # Load the cascade model from CV2
            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades +
                'haarcascade_frontalface_default.xml')
        return self._face_detector

    @property
    def face_recognizer(self):
        if self._face_recognizer is None:
            self._load_trained_models()
        return self._face_recognizer

    @property
    def ids_to_names(self):
        if self._ids_to_names is None:
            self._load_trained_models()
        return self._ids_to_names

    @property
    def num_training_examples(self):
        if self._num_training_examples is None:
            self._load_trained_models()
        return self._num_training_examples

    def _load_trained_models(self):
        if not os.path.exists(model_metadata_file_path):
            print('Failed to find: ' + model_metadata_file_path)
            return

        with open(model_metadata_file_path, 'rb') as fh:
            model_metadata = yaml.load(fh, Loader=yaml.Loader)

        self._ids_to_names = model_metadata['ids_to_names']
        self._num_training_examples = Counter(model_metadata['num_training_examples'])

        self._face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(face_model_path):
            print('Did not get face model file! ' + face_model_path)
            return
        self._face_recognizer.read(face_model_path)

    def get_description(self):
        if not self.num_training_examples:
            return '<li> No faces learned yet!  You can make a model using the Colab Notebook.</li>'
        recs = []
        for name, samples in self.num_training_examples.items():
            recs.append('<li> {name} learned from {samples}.'.format(
                name=name, samples=samples))
        return ''.join(recs)


# make a global variable instance of Models
models = Models()


def get_image():
    'take a picture with the camera return in numpy array format'
    with picamera.PiCamera() as camera:
        camera.start_preview()
        time.sleep(2)
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            return stream.array
        
        
def get_labeled_image():
    'take picture, recognize faces, insert text labels, save to file'

    if models.face_detector is None or models.face_recognizer is None:
        return 

    img = get_image()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = models.face_detector.detectMultiScale(
        image=gray_image,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30,30),
    )
    
    for face_idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = gray_image[y:y + h, x:x + w]
        face_id, inverted_confidence = \
            models.face_recognizer.predict(face_image)
        confidence = 100 - inverted_confidence
        if (confidence <= 0):
            name = '(unknown)'
        else:
            name = models.ids_to_names[face_id]

        mesg = '{0} {1}%'.format(name, round(confidence))
        cv2.putText(img, mesg,
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36,255,12), 2)

    print('trying to save to ' + current_image_file_path)
    imsave(current_image_file_path, img)


def index():
    'take a new picture, generate HTML page, report model stats'
    
    get_labeled_image()

    known_faces = models.get_description()
    
    html = u'''<html>
  <body>
    <h1>Face Identifier</h1>
      <img src="/{current_image}" height="300px" width="300px"/>

    <h1>Model</h1>
    <p>Currently, the model knows these people:
    <ul>{faces}</ul>

    <h1>Update Model</h1>

    To update the model, please uload both <b>model_metadata.yml</b>
    and <b>face_identification_model.yml</b>: <br/>

    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" multiple/>
      <input type="submit" value="upload">
    </form>
  </body>
</html>
'''.format(current_image=current_image_name, faces=known_faces)
    return html


class FaceIdentifier(http.server.BaseHTTPRequestHandler):
    '''FaceIdentifier handler provides a simple browser-based interface to
    face recognition models and the Raspberry Pi's camera.

    '''
    
    def do_GET(self):
        '''Respond to all GET requests with the same HTML page unless the path
        is specifically for "/current_image.jpg" which returns the
        image with face identification information added into the
        pixels.

        '''
    
        if self.path == '/' + current_image_name:
            if not os.path.exists(current_image_file_path):
                data = None
            else:
                with open(current_image_file_path, 'rb') as fh:
                    data = fh.read()
            if data:
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.send_header("Content-type", "image/jpeg")
                self.send_header("Content-Length", 0)
                self.end_headers()

        else:
            '''use the `index` function to generate HTML for the main page.'''
            response = index()
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", len(response))
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        '''Handle post requests by saving files in the local dir.

        This is needed for uploading the model files created by the
        notebook.

        '''
        ret, info = self.deal_post_data()
        print((ret, info, "by: ", self.client_address))
        success_or_fail = ret and 'success' or 'fail'
        html = u'''<html>
<title>Upload Result Page</title>
<body>
<h2>Upload Result Page</h2>
<strong>{success_or_fail}</strong>
<br/>
{info}
<br/>
<a href="/">back</a>
'''.format(success_or_fail=success_or_fail, info=info)
        
        if ret:
            # if we got files, clear the old model data
            models.clear()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html.encode())

    def deal_post_data(self):
        'figure out file names from multi-file upload'

        uploaded_files = []
        content_type = self.headers['content-type']
        if not content_type:
            return (False, "Content-Type header doesn't contain boundary")
        boundary = content_type.split("=")[1].encode()
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (False, "Content NOT begin with boundary")
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line.decode())
            if not fn:
                return (False, "Can't find out file name...")
            path = this_dir   #os.getcwd()
            fn = os.path.join(path, fn[0])
            line = self.rfile.readline()
            remainbytes -= len(line)
            line = self.rfile.readline()
            remainbytes -= len(line)
            try:
                out = open(fn, 'wb')
            except IOError as exc:
                print(traceback.format_exc(exc))
                return (False, "Can't create file to write, do you have permission to write?")
            else:
                with out:                    
                    preline = self.rfile.readline()
                    remainbytes -= len(preline)
                    while remainbytes > 0:
                        line = self.rfile.readline()
                        remainbytes -= len(line)
                        if boundary in line:
                            preline = preline[0:-1]
                            if preline.endswith(b'\r'):
                                preline = preline[0:-1]
                            out.write(preline)
                            uploaded_files.append(fn)
                            break
                        else:
                            out.write(preline)
                            preline = line
        return (True, "File(s) <br/>%s<br/> upload success!" % "<br/>".join(uploaded_files))


            
def run(server_class=http.server.HTTPServer, handler_class=FaceIdentifier):
    'launch the FaceIdentifier handler with serve_forever'
    server_address = ('', 80)
    print('Serving forever...')
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    elif args.test:
        print(index())


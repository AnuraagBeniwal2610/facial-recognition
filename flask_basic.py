from flask import Flask
from flask import request, jsonify
import cv2

app = Flask(__name__)

@app.route('/LiveCapture', methods=['POST'])
def capture():
	videoCaptureObject = cv2.VideoCapture(0)
	result = True
	while(result):
	    ret,frame = videoCaptureObject.read()
	    cv2.imwrite("NewPicture.jpg",frame)
	    result = False
	videoCaptureObject.release()
	cv2.destroyAllWindows()
	return "image captured"

app.run(debug=True)


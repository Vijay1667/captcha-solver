import cv2
import os
from flask import Flask
from flask_cors import CORS
from flask import Flask,jsonify
from flask import request
import cv2
import numpy as np
import base64
os.environ['KERAS_BACKEND'] = 'theano'
app = Flask(__name__)
CORS(app)
# from google.colab.patches import cv2_imshow
file_list = os.listdir('./dataset/')
# image2 = cv2.imread("download.png")
image_files = [file for file in file_list if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
aton = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0,0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
charmap = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,  'P': 24, 'Q': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29, 'V': 30, 'W': 31, 'X': 32, 'Y': 33, 'Z': 34}
train_images=[]
outputs=[]
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join('./dataset/', image_file)
    filename, extension = os.path.splitext(image_file)
    for char in filename:
      if char in aton:
        outputs.append(charmap[char])
        # print(char,">>",charmap[char])
        aton[char] += 1
      elif int(char) in aton:  # Check if the character is a digit
        outputs.append(charmap[int(char)])
        # print(int(char),">>",charmap[int(char)])
        aton[int(char)] += 1
    # Read the image using OpenCV
    image2 = cv2.imread(image_path)
    ret,image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow(image2)
    height,width,channels=image2.shape
    height1=width//6
    init=0
    for i in range(0,6):
        part=image2[:,init:height1]
        # print(part.shape)
        gray_image = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        np.set_printoptions(threshold=np.inf)

        # print([np.reshape(np.array(gray_image),(1350,))])
        train_images.append(np.reshape(np.array(gray_image),(1350,)))
        # res=model.predict(np.array([np.reshape(np.array(gray_image),(1350,))]))
        # print((np.argmax(res[0])))
        # print(mapping.get(np.argmax(res[0])))
        init=height1
        height1=height1+(width//6)
        #cv2.imshow(gray_image)

# print(outputs)

from keras.utils import to_categorical

encoded_labels = to_categorical(outputs)
# print(encoded_labels)
print(len(charmap))
print(len(aton))
print(aton)

from keras import models
from keras import layers
from keras import Sequential
from keras.layers import Dense
model=Sequential([Dense(1024,activation='relu',input_shape=(1350,)),Dense(512,activation='relu'),Dense(256,activation='relu'),Dense(35,activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(np.array(train_images),encoded_labels,epochs=100)

# import cv2
# import numpy as np
# import os
# from google.colab.patches import cv2.imshow
file_list = os.listdir('./testing/')
# image2 = cv2.imread("download.png")
image_files_test = [file for file in file_list if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

test_images=[]
test_outputs=[]
for image_file_test in image_files_test:
    # Construct the full path to the image
    image_path = os.path.join('./testing/', image_file_test)
    filename, extension = os.path.splitext(image_file_test)
    for char in filename:
      if char in aton:
        test_outputs.append(charmap[char])
        aton[char] += 1
      elif int(char) in aton:  # Check if the character is a digit
        test_outputs.append(charmap[int(char)])
        aton[int(char)] += 1
    # Read the image using OpenCV
    image2 = cv2.imread(image_path)
    ret,image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow(image2)
    height,width,channels=image2.shape
    height1=width//6;
    init=0
    captcha=""
    for i in range(0,6):
        part=image2[:,init:height1]
        # print(part.shape)
        gray_image = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        np.set_printoptions(threshold=np.inf)

        # print([np.reshape(np.array(gray_image),(1350,))])
        test_images.append(np.reshape(np.array(gray_image),(1350,)))
        res=model.predict(np.array([np.reshape(np.array(gray_image),(1350,))]))
        # print(res)
        softmax_sum = np.sum(res[0])

        # print("Sum of softmax outputs:", softmax_sum)
        # print("Sum of softmax outputs:", np.argmax(res[0]))
        for key, value in charmap.items():
          if value == np.argmax(res[0]):
            captcha=captcha+str(key)
            # print(f"Key with value {np.argmax(res[0])} found: {key}")
            break
            # print(f"No key found with value {np.argmax(res[0])}")
        # res=model.predict(np.array([np.reshape(np.array(gray_image),(1350,))]))
        # print((np.argmax(res[0])))
        # print(mapping.get(np.argmax(res[0])))
        init=height1
        height1=height1+(width//6)
        # cv2.imshow(gray_image)
    print(captcha)


@app.route('/evalcaptcha',methods=['POST'])
def about():
    data=request.get_json()
    captcha = data.get('captcha', {})
    
    # Your base64-encoded image string
    base64_image_string = captcha
    
    # Extract the base64 part
    encoded_data = base64_image_string.split(',')[1]
    
    # Decode the base64 string into bytes
    image_data = base64.b64decode(encoded_data)
    
    # Convert the bytes to a NumPy array
    image_np_array = np.frombuffer(image_data, dtype=np.uint8)
    
    # Read the image using cv2.imdecode
    image2 = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    ret,image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Decoded Image', image2)
    # cv2.waitKey(0)

    height,width,channels=image2.shape
    height1=width//6;
    init=0
    captcha=""
    for i in range(0,6):
        part=image2[:,init:height1]
        # print(part.shape)
        gray_image = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        np.set_printoptions(threshold=np.inf)

        # print([np.reshape(np.array(gray_image),(1350,))])
        test_images.append(np.reshape(np.array(gray_image),(1350,)))
        res=model.predict(np.array([np.reshape(np.array(gray_image),(1350,))]))
        # print(res)
        softmax_sum = np.sum(res[0])

        # print("Sum of softmax outputs:", softmax_sum)
        # print("Sum of softmax outputs:", np.argmax(res[0]))
        for key, value in charmap.items():
          if value == np.argmax(res[0]):
            captcha=captcha+str(key)
            # print(f"Key with value {np.argmax(res[0])} found: {key}")
            break
            # print(f"No key found with value {np.argmax(res[0])}")
        # res=model.predict(np.array([np.reshape(np.array(gray_image),(1350,))]))
        # print((np.argmax(res[0])))
        # print(mapping.get(np.argmax(res[0])))
        init=height1
        height1=height1+(width//6)
        # cv2.imshow(gray_image)
    print(captcha)
    # Now, 'image' contains the image data
    # You can use cv2.imshow or process the image further
    # cv2.imshow('Decoded Image', image)
    # cv2.waitKey(0)
    return captcha

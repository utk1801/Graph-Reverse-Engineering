# ==========================================
#           Importing modules
# ==========================================
from flask_cors import CORS, cross_origin
import boto3
from flask import Flask, jsonify, Response, render_template, request, send_file

import pandas as pd
import numpy as np
import csv
import json
import sys
import os
from werkzeug.utils import secure_filename

from PIL import Image, ImageColor
import operator
import math

from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import collections

from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
import warnings
warnings.filterwarnings("ignore")
from config import ACCESS_KEY,SECRET_KEY

data_ink_ratio = None
background_shade = None

app = Flask(__name__)
cors = CORS(app)

app.secret_key = 'some_random_key'
bucket_name = "523testing"
app.config['CORS_HEADERS'] = 'Content-Type'

s3 = boto3.client(
   "s3",
   aws_access_key_id=ACCESS_KEY,
   aws_secret_access_key=SECRET_KEY
   )
bucket_resource = s3
# ==========================================
#           Load particular files
# ==========================================
file_path = './data/features_academic.csv'
df = pd.read_csv(file_path)

# ==========================================
#              Label encoding  
# ==========================================
label_encoding = preprocessing.LabelEncoder()
l = label_encoding.fit(df['type'])
label = label_encoding.transform(df['type'])
df['type_encoded'] = label

# ==========================================
#          Labels Dictionary
# ==========================================
label_encoding_dict = dict({})

for index, row in df.iterrows():
  if df.at[index, 'type_encoded'] not in label_encoding_dict:
    label_encoding_dict[df.at[index, 'type_encoded']] = df.at[index, 'type']

for key, value in sorted(label_encoding_dict.items()):
  print(key, value)

# ==========================================
#        Prepare Train & Test Data
# ==========================================
x_vector = ['fw',
            'fh',
            'x',
            'y',
            'x2',
            'y2',
            'w',
            'h'
            ]

index_cut = 5900
x_train = df[x_vector][:index_cut]
y_train = df[['type_encoded']][:index_cut]

x_test = df[x_vector][index_cut:]
y_test = df[['type_encoded']][index_cut:]


# ==========================================
#          Select Best Model
# ==========================================
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred_r = regr.predict(x_test)
m1 = mean_absolute_error(y_test,y_pred_r)
print("MSE for test data [Linear Regression]: %.5f" % m1)

regr2 = KNeighborsRegressor(n_neighbors=3)
regr2.fit(x_train, y_train)
y_pred_r_2 = regr2.predict(x_test)
m2 = mean_absolute_error(y_test,y_pred_r_2)
print("MSE for test data [KNeighbors Regression]: %.5f" % m2)

regr3 = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
regr3.fit(x_train, y_train)
y_pred_r_3 = regr3.predict(x_test)
m3 = mean_absolute_error(y_test,y_pred_r_3)
print("MSE for test data [AdaBoost Regression]: %.5f" % m3)

regr4 = RandomForestRegressor(max_depth=2, random_state=0)
regr4.fit(x_train, y_train)
y_pred_r_4 = regr4.predict(x_test)
m4 = mean_absolute_error(y_test,y_pred_r_4)
print("MSE for test data [RandomForest Regression]: %.5f" % m4)

best_model = ''
if m1 < m2 and m1 < m3 and m1 < m4:
  regr = linear_model.LinearRegression()
  best_model = 'Linear Regression'

elif m2 < m1 and m2 < m3 and m2 < m4:
  regr = KNeighborsRegressor(n_neighbors=3)
  best_model = 'K Neighbors Regression'
  
elif m3 < m1 and m3 < m2 and m3 < m4:
  regr = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
  best_model = 'AdaBoost Regression'

elif m4 < m1 and m4 < m2 and m4 < m3:
  regr = RandomForestRegressor(max_depth=2, random_state=0)
  best_model = 'Randon Forest Regression'
  
print('Best Model:', best_model)

regr2 = KNeighborsRegressor(n_neighbors=1)
regr2.fit(x_train, y_train)
y_pred_r_2 = regr2.predict(x_test)
m2 = mean_absolute_error(y_test, y_pred_r_2)
print("MSE for test data: %.5f" % m2)
print("Accuracy: %.5f" % (accuracy_score(y_test, y_pred_r_2) * 100), '%')

data_ink_map = list([])
black_ink_added = False
chart_w = chart_h = 0

def calculate_data_ink(img_path):

  global data_ink_map
  global black_ink_added
  global chart_w
  global chart_h

  im = Image.open(img_path).convert('RGB')
  chart_w, chart_h = im.size

  def rgb_to_hex(_r, _g, _b):
      return '#{:02x}{:02x}{:02x}'.format(_r, _g, _b)

  def isLightOrDark(rgbColor):
      [r,g,b]=rgbColor
      hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
      if (hsp>250):
          return 'light'
      else:
          return 'dark'

  def sort_map(_map):
    sorted_d = dict(sorted(_map.items(), key = operator.itemgetter(1), reverse=True))
    return sorted_d

  def print_map(_map):

    cut_off = 11
    count = 0
    global data_ink_map
    global black_ink_added

    print('\t<<< Top Shades >>>>')
    print("{:<15} {:<15} {:<15}".format('Color', 'Frequency', 'Shade')) 
    for key, value in _map.items():
      RGB = ImageColor.getrgb(str(key))
      shade = isLightOrDark(list(RGB))

      data_ink_map.append((key, value, shade))
      print("{:<15} {:<15} {:<15}".format(key, value, shade))

      if key == '#000000':
        black_ink_added = True

      if count > cut_off:
        break
      count += 1

  def generate_color_map(im):

    hex_color_map = dict({})
    rows, cols = im.size

    for i in range(rows):
      for j in range(cols):
        _r, _g, _b = im.getpixel((i, j))
        hex_code = rgb_to_hex(_r, _g, _b)

        if hex_code not in hex_color_map:
          hex_color_map[hex_code] = 1
        else:
          hex_color_map[hex_code] += 1

    return hex_color_map

  hex_color_map = generate_color_map(im)
  hex_color_map = sort_map(hex_color_map)
  print_map(hex_color_map)

  # calculate data ink
  data_pixels = 0
  for key, value, shade in data_ink_map:
    if shade == 'dark':
      data_pixels += value

  # add black pixels if not added
  if not black_ink_added:

    if '#000000' in hex_color_map:
      data_pixels += hex_color_map['#000000']
    else:
      data_pixels += 0

  print('\nTotal pixels of data:', data_pixels)
  print('Total pixels in chart:', sum(hex_color_map.values()))
  print('Data-ink ratio:', round(data_pixels/sum(hex_color_map.values()), 4))
  return round(data_pixels/sum(hex_color_map.values()), 4), data_ink_map[0][2]


def testing_aws_and_model(aws_data):

  global chart_h
  global chart_w

  print(chart_w, chart_h)

  # # Opening JSON file 
  # f = open(json_path) 
    
  # # returns JSON object as  
  # # a dictionary 
  # _data = json.load(f) 
    
  # # Closing file 
  # f.close() 

  _data = aws_data

  chart_width = chart_w
  chart_height = chart_h
  _all_ftrs = []
  _all_ftrs_for_training = []

  data = {
      'available': [],
      'not_available': [],
      'x_spread_ratio': 0,
      'y_spread_ratio': 0,
      '_all_boxes': []
  }

  TextDetections = _data['TextDetections']
  for text in TextDetections:


    if text['Type'] == 'WORD':

      # print(text)
      box = dict({})

      box['fw'] = chart_width
      box['fh'] = chart_height

      _all_x = []
      _all_y = []

      for i in range(4):
        _all_x.append(round(float(text['Geometry']['Polygon'][i]['X']) * chart_width))
        _all_y.append(round(float(text['Geometry']['Polygon'][i]['Y']) * chart_height))
      
      # top-left
      box['x'] = min(_all_x)
      box['y'] = min(_all_y)

      # bottom-right
      box['x2'] = max(_all_x)
      box['y2'] = max(_all_y)

      box['w'] = max(_all_x) - min(_all_x)
      box['h'] = max(_all_y) - min(_all_y)

      box['text'] = text['DetectedText']
      
      _all_ftrs.append(list(box.values()))
      collect_numerical_ftrs = []
      for key, val in box.items():
        if type(val) == int:
          collect_numerical_ftrs.append(val)
      _all_ftrs_for_training.append(collect_numerical_ftrs)

      data['_all_boxes'].append([box['x'], box['y'], box['w'], box['h']])

  df_test = pd.DataFrame(_all_ftrs_for_training, columns = ['fw', 'fh', 'x', 'y', 'x2', 'y2', 'w', 'h'])  

  regr2 = KNeighborsRegressor(n_neighbors=1)
  regr2.fit(x_train, y_train)
  y_pred_r_2 = regr2.predict(df_test)

  text_and_class = collections.defaultdict(list)
  for index, pred_lb in enumerate(y_pred_r_2.tolist()):
    text_and_class[label_encoding_dict[int(pred_lb[0])]].append(_all_ftrs[index][-1])
  print(text_and_class)

  obtained_ftrs_set = set()
  for each in y_pred_r_2.tolist():
    if int(each[0]) not in obtained_ftrs_set:
      obtained_ftrs_set.add(int(each[0]))

  x_axis_lb_code = x_axis_lb_code = -1
  print('{:<15} {:<15}'.format('Property', 'Presence\n'))
  for key, value in sorted(label_encoding_dict.items()):
    
    if key not in obtained_ftrs_set:
      print('{:<15} {:<15}'.format(value, 'Missing'))
      data['not_available'].append(value)
    else:
      print('{:<15} {:<15}'.format(value, 'Available'))
      data['available'].append(value)

    if value == 'x-axis-label':
      x_axis_lb_code = key

    if value == 'y-axis-label':
      y_axis_lb_code = key

  y_pred_r_2 = pd.DataFrame(y_pred_r_2, columns = ['pred'])
  res_df = pd.concat([df_test, y_pred_r_2], axis= 1)

  x_spread = y_spread = 0
  for index, row in res_df.iterrows():
    if int(row['pred']) == x_axis_lb_code:
      x_spread += int(row['w'])
    
    if int(row['pred']) == y_axis_lb_code:
      y_spread += int(row['h'])

  data['x_spread_ratio'] = x_spread/chart_w
  data['y_spread_ratio'] = y_spread/chart_h

  data['text_and_class'] = text_and_class

  return data

def upload_file(file_name, bucket):
    """
    Function to upload a file to an S3 bucket
    """
    object_name = file_name
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(file_name, bucket, object_name)

    return response

@app.route('/')
def template():
    return render_template('index.html')

@app.route('/chart_data', methods=['POST'])
def chart_data(_data):
  
        data = testing_aws_and_model(_data)

        data['data_ink_ratio_value'] = data_ink_ratio

        if data_ink_ratio < 0.05:
          data['data_ink_ratio_score'] = 25
          data['data_ink_ratio_comment'] = "Data-ink ratio is low. Consider adding more valid data."
        elif data_ink_ratio >= 0.05 and data_ink_ratio < 0.08:
          data['data_ink_ratio_score'] = 50
          data['data_ink_ratio_comment'] = "Data-ink ratio is moderate. Consider adding more valid data."
        elif data_ink_ratio >= 0.08 and data_ink_ratio < 0.15:
          data['data_ink_ratio_score'] = 75
          data['data_ink_ratio_comment'] = "Data-ink ratio is good."
        elif data_ink_ratio >= 0.15:
          data['data_ink_ratio_score'] = 100
          data['data_ink_ratio_comment'] = "Data-ink is perfect!"

        data['chart_elem_score'] = round((len(data['available'])/(len(data['available']) + len(data['not_available']))) * 100)
        
        x_spread_ratio = data['x_spread_ratio']
        y_spread_ratio = data['y_spread_ratio']

        if x_spread_ratio < 0.1:
          data['x_spread_ratio_score'] = 25
          data['x_spread_ratio_comment'] = "X-axis not spread or scaled properly."
        elif x_spread_ratio >= 0.1 and x_spread_ratio < 0.2:
          data['x_spread_ratio_score'] = 50
          data['x_spread_ratio_comment'] = "X-axis might not spread or scaled properly."
        elif x_spread_ratio >= 0.2 and x_spread_ratio < 0.3:
          data['x_spread_ratio_score'] = 75
          data['x_spread_ratio_comment'] = "X-axis seems good."
        elif x_spread_ratio >= 0.3:
          data['x_spread_ratio_score'] = 100
          data['x_spread_ratio_comment'] = "X-axis is perfect!"

        if y_spread_ratio < 0.05:
          data['y_spread_ratio_score'] = 25
          data['y_spread_ratio_comment'] = "Y-axis not spread or scaled properly."
        elif y_spread_ratio >= 0.05 and y_spread_ratio < 0.1:
          data['y_spread_ratio_score'] = 50
          data['y_spread_ratio_comment'] = "Y-axis might not spread or scaled properly."
        elif y_spread_ratio >= 0.1 and y_spread_ratio < 0.2:
          data['y_spread_ratio_score'] = 75
          data['y_spread_ratio_comment'] = "Y-axis seems good."
        elif y_spread_ratio >= 0.2:
          data['y_spread_ratio_score'] = 100
          data['y_spread_ratio_comment'] = "Y-axis is perfect!"

        if background_shade == 'light':
          data['background_score'] = 100
          data['background_score_comment'] = "Good background choice."
        else:
          data['background_score'] = 50
          data['background_score_comment'] = "Poor background choice."

        data['spacing_score'] = (data['x_spread_ratio_score'] + data['y_spread_ratio_score']) / 2

        _all_score = data['data_ink_ratio_score'] + ((data['x_spread_ratio_score'] + data['y_spread_ratio_score']) / 2) + data['background_score'] + data['chart_elem_score']
        data['overall_score'] = round((_all_score / 400) * 100)

        """
        Prepare entence for polly
        """

        data["speak"] = "Let's start with data ink ratio, the value is " + str(data['data_ink_ratio_score']) + ". This is a bar chart with 4 bars arranged horizontally. The data ink ratio is 70% and the background is white which looks good. Most chart elements are present but the title is missing. I would suggest adding a title to this chart to make it more instructive to someone who does not know what the chart represents."

        return data

  
@app.route('/save_local', methods=['POST'])
def upload_local():

       global data_ink_ratio
       global background_shade

       file = request.files['img']
       filename = secure_filename(file.filename)
       file.save(os.path.join('./static/images', filename))

       data_ink_ratio, background_shade = calculate_data_ink('./static/images/' + filename)
       print(data_ink_ratio, background_shade)

       return 'saved local'

@app.route('/upload', methods=['POST'])
def upload():
       if request.method == "POST":
        try:
            img = request.files['img']
            filename = ''
            if img:
                filename = secure_filename(img.filename)
                img.save(filename)
                bucket_resource.upload_file(
                    Bucket = bucket_name,
                    Filename=filename,
                    Key=filename
                )
                return("<h1>upload successful<h1>")
        except Exception as e:
            return (str(e))
        return render_template("index.html")

@app.route('/recog', methods=['POST'])
def lambda_handler():
      try:
        file = request.files['img']
        filename = secure_filename(file.filename)  
        bucket='523testing'

        s3_rekog=boto3.client(
          "rekognition",  region_name='us-east-1',
          aws_access_key_id=ACCESS_KEY,
          aws_secret_access_key=SECRET_KEY
          )
        
        text=s3_rekog.detect_text(Image={
          'S3Object': 
          {'Bucket':bucket,'Name': filename
          } 
        })
        res = {
          "textFound": text
        }

      except Exception as e:
          return (str(e))
      return json.dumps(chart_data(text))

AUDIO_FORMATS = {"ogg_vorbis": "audio/ogg",
                 "mp3": "audio/mpeg",
                 "pcm": "audio/wave; codecs=1"}

# Create a client using the credentials and region defined in the adminuser
# section of the AWS credentials and configuration files
session = Session(profile_name="default")
polly = session.client("polly")

# # Create a flask app
# app = Flask(__name__)


# Simple exception class
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


# Register error handler
@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/read', methods=['GET'])
def read():
    """Handles routing for reading text (speech synthesis)"""
    # Get the parameters from the query string
    try:
        outputFormat = request.args.get('outputFormat')
        text = request.args.get('text')
        voiceId = request.args.get('voiceId')
    except TypeError:
        raise InvalidUsage("Wrong parameters", status_code=400)

    # Validate the parameters, set error flag in case of unexpected
    # values
    if len(text) == 0 or len(voiceId) == 0 or \
            outputFormat not in AUDIO_FORMATS:
        raise InvalidUsage("Wrong parameters", status_code=400)
    else:
        try:
            # Request speech synthesis
            response = polly.synthesize_speech(Text=text,
                                               VoiceId=voiceId,
                                               OutputFormat=outputFormat)
        except (BotoCoreError, ClientError) as err:
            # The service returned an error
            raise InvalidUsage(str(err), status_code=500)

        return send_file(response.get("AudioStream"),
                         AUDIO_FORMATS[outputFormat])


@app.route('/voices', methods=['GET'])
def voices():
    """Handles routing for listing available voices"""
    params = {}
    voices = []

    try:
        # Request list of available voices, if a continuation token
        # was returned by the previous call then use it to continue
        # listing
        response = polly.describe_voices(**params)
    except (BotoCoreError, ClientError) as err:
        # The service returned an error
        raise InvalidUsage(str(err), status_code=500)

    # Collect all the voices
    voices.extend(response.get("Voices", []))

    return jsonify(voices)

if __name__ == '__main__':

    app.run('localhost', 8081)
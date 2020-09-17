# ==========================================
#           Importing modules
# ==========================================
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import boto3

import pandas as pd
import numpy as np
import csv
import json
import sys
import os
from werkzeug.utils import secure_filename

# from PIL import Image, ImageColor
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
from config import ACCESS_KEY,SECRET_KEY

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

app.secret_key = 'some_random_key'
bucket_name = "523-testimg"

s3 = boto3.client(
   "s3",
   aws_access_key_id=ACCESS_KEY,
   aws_secret_access_key=SECRET_KEY
)
bucket_resource = s3


# original_data = pandas.read_csv('players_2020_sample.csv')
# original_data = original_data.fillna(0)

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


def testing_aws_and_model(json_path):

  global chart_h
  global chart_w

  print(chart_w, chart_h)

  # Opening JSON file 
  f = open(json_path) 
    
  # returns JSON object as  
  # a dictionary 
  _data = json.load(f) 
    
  # Closing file 
  f.close() 

  chart_width = chart_w
  chart_height = chart_h
  _all_ftrs = []

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

      _all_ftrs.append(list(box.values()))


  df_test = pd.DataFrame(_all_ftrs, columns = ['fw', 'fh', 'x', 'y', 'x2', 'y2', 'w', 'h'])  

  regr2 = KNeighborsRegressor(n_neighbors=1)
  regr2.fit(x_train, y_train)
  y_pred_r_2 = regr2.predict(df_test)

  data = {
      'available': [],
      'not_available': [],
      'x_spread_ratio': 0,
      'y_spread_ratio': 0
  }
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
def chart_data():
    try:
        file = request.files['myFile']
        filename = secure_filename(file.filename)
        file.save(os.path.join('./static/images', filename))
        # req_data = request.get_json()

        data_ink_ratio, background_shade = calculate_data_ink('./static/images/' + filename)
        data = testing_aws_and_model('./data/testing_charts/test_01.json')

        data['data_ink_ratio_value'] = data_ink_ratio

        if data_ink_ratio < 0.05:
          data['data_ink_ratio_score'] = 25
          data['data_ink_ratio_comment']="This needs some redesign and doesn't qualify well on Tufte's scale of Goodness of Fit. Can do better! :)"
        elif data_ink_ratio >= 0.05 and data_ink_ratio < 0.08:
          data['data_ink_ratio_score'] = 50
          data['data_ink_ratio_comment']="This is a decent graph, with Data Ink ratio lying in the range of 5-8%. The aesthetics can be improved by "
        elif data_ink_ratio >= 0.08 and data_ink_ratio < 0.15:
          data['data_ink_ratio_score'] = 75
        elif data_ink_ratio >= 0.15:
          data['data_ink_ratio_score'] = 100

        data['chart_elem_score'] = round((len(data['available'])/(len(data['available']) + len(data['not_available']))) * 100)
        
        x_spread_ratio = data['x_spread_ratio']
        y_spread_ratio = data['y_spread_ratio']

        if x_spread_ratio < 0.1:
          data['x_spread_ratio_score'] = 25
        elif x_spread_ratio >= 0.1 and x_spread_ratio < 0.2:
          data['x_spread_ratio_score'] = 50
        elif x_spread_ratio >= 0.2 and x_spread_ratio < 0.3:
          data['x_spread_ratio_score'] = 75
        elif x_spread_ratio >= 0.3:
          data['x_spread_ratio_score'] = 100

        if y_spread_ratio < 0.05:
          data['y_spread_ratio_score'] = 25
        elif y_spread_ratio >= 0.05 and y_spread_ratio < 0.1:
          data['y_spread_ratio_score'] = 50
        elif y_spread_ratio >= 0.1 and y_spread_ratio < 0.2:
          data['y_spread_ratio_score'] = 75
        elif y_spread_ratio >= 0.2:
          data['y_spread_ratio_score'] = 100

        if background_shade == 'light':
          data['background_score'] = 100
        else:
          data['background_score'] = 50

        data['spacing_score'] = (data['x_spread_ratio_score'] + data['y_spread_ratio_score']) / 2

        _all_score = data['data_ink_ratio_score'] + ((data['x_spread_ratio_score'] + data['y_spread_ratio_score']) / 2) + data['background_score'] + data['chart_elem_score']

        data['overall_score'] = round((_all_score / 400) * 100)

    except:
        e = sys.exc_info()
    return json.dumps(data)

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

if __name__ == '__main__':
    app.run('localhost', 8081)
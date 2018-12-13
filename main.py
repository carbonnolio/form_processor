from FormToImage import ImagePreprocessor
from FormTransform import FormTransform
from GridTools import GridTools

import cv2
import base64
import os
from PIL import Image
import json
import urllib.request
import requests


dict_path = ''
sample_path = ''

forms_dir = ''
one_file_dir = ''

processor = ImagePreprocessor(is_debug=True, form_dict_path=dict_path)
trans = FormTransform(1000, 0.15, is_debug=True)
grid_tools = GridTools(is_debug=True)

sample_res = processor.to_normal_type(sample_path)

idx = 0
form_list = []

url = ''

def send_request(baseUrl, file_name, img, payload):
    files = {
         'json': (None, json.dumps(payload), 'application/json'),
         'file': (file_name, img, 'application/octet-stream')
    }

    url = baseUrl + '?' + 'filename=' + file_name + '&featurename=' + payload['featureType']

    r = requests.post(url, files=files)
    print(r.content)


# json_data_as_bytes = json_data.encode('utf-8')
# req.add_header('Content-Length', len(json_data))
# print (json_data)
# response = urllib.request.urlopen(req, json_data)

files = os.listdir(one_file_dir)

for file_name in files:
    file_path = forms_dir + '/' + file_name
    
    img_forms = processor.to_normal_type(file_path)
# len(img_forms)
    for i in range(0, 1):
        form_name = '{}_{}'.format(i, file_name)

        try:
            # form_data = {}
            # form_data['formName'] = form_name

            form_type, form_type_res = processor.detect_form_type(img_forms[i])
            # form_data['formType'] = form_type

            # form_features = []

            corrected = trans.align_image(form_type_res, sample_res[0])
            features = processor.get_grid_fields(corrected, form_type=form_type)

            for feature_type, img in features:

                form_data = {}
                form_data['formName'] = form_name
                form_data['formType'] = form_type
                form_data['featureType'] = feature_type


                # feature_data = {}
                # feature_data['featureType'] = feature_type

                # Clearing grid
                clear_img = grid_tools.remove_grid(img, str(idx))

                idx += 1

                # _, buffer = cv2.imencode('.png', img)

                send_request(url, form_name, clear_img, form_data)

                # feature_data['featureImg'] = base64.b64encode(buffer)

            #     form_features.append(feature_data)

            # form_data['features'] = form_features

            # form_list.append(form_data)
        except:
            print(form_name)
            raise
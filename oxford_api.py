import json
import os
import http.client, urllib.request, urllib.parse, urllib.error, base64
import sub_key

json_cache_path = 'json_cache'

# API vars
headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': sub_key.get_key(),
}

params = urllib.parse.urlencode({
    # Request parameters
    'language': 'unk',
    'detectOrientation ': 'true',
})

def get_json_data(image, base_path, zoom_level, pref):
  zoom_prefix = str(zoom_level) + 'x/' if zoom_level > 1 else ''
  json_cache_file = pref + json_cache_path + '/' + zoom_prefix + image + '.json'

  if os.path.isfile(json_cache_file):
    data = json.loads(open(json_cache_file, 'r').read())

    if 'statusCode' not in data or data['statusCode'] != 429:
      return data

  img_data = open(base_path + '/' + zoom_prefix + image, 'rb').read()

  try:
    conn = http.client.HTTPSConnection('api.projectoxford.ai')
    conn.request("POST", "/vision/v1/ocr?%s" % params, img_data, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
  except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

  json_data = json.loads(data.decode('utf-8')) # Need to double-check if utf-8 is correct

  with open(json_cache_file, 'w') as json_file:
    json.dump(json_data, json_file)

  return json_data

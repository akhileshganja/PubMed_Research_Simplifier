import urllib.request, json
import urllib.error

data = json.dumps({'query': 'cancer'}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/search', data=data, headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print("ERROR BODY:", e.read().decode('utf-8'))

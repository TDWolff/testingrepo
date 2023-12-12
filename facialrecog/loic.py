import requests

for _ in range(1):  # repeat 10 times
    with open('/Users/torinwolff/Documents/GitHub/testingrepo/facialrecog/loic.txt', 'rb') as file:
        response = requests.post('https://go.gopwin24.com/1kgD41lYxGk', files={'file': file})
    print(response.status_code)
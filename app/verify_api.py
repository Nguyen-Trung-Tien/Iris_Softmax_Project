import requests
import json
import sys

def test_api():
    url = "http://127.0.0.1:5000/api/predict"
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    print(f"Testing API endpoint: {url}")
    print(f"Payload: {data}")
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            
            # Check for expected keys
            resp_json = response.json()
            if "prediction" in resp_json and "probabilities" in resp_json:
                print("SUCCESS: API returned prediction and probabilities.")
            else:
                print("FAILURE: API did not return expected keys.")
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")
        print("Ensure the Flask app is running!")

def test_csv_prediction():
    url = "http://127.0.0.1:5000/predict_csv"
    # Create a dummy CSV content
    csv_content = "sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n6.0,3.0,4.8,1.8"
    files = {'file': ('test.csv', csv_content, 'text/csv')}
    
    print(f"\nTesting CSV Endpoint: {url}")
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            if "Analytics Dashboard" in response.text:
                print("SUCCESS: CSV analysis returned dashboard.")
            else:
                print("FAILURE: Dashboard text not found in response.")
        else:
            print("FAILURE: Endpoint returned error.")
    except Exception as e:
        print(f"CSV Test Failed: {e}")

def test_about_page():
    url = "http://127.0.0.1:5000/about"
    print(f"\nTesting About Page: {url}")
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: About page loaded.")
        else:
            print("FAILURE: About page returned error.")
    except Exception as e:
        print(f"About Page Test Failed: {e}")

if __name__ == "__main__":
    test_api()
    test_csv_prediction()
    test_about_page()

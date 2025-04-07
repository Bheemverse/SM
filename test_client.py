import requests
import json

def test_get_rules():
    # Test getting rules with default parameters
    response = requests.get('http://localhost:5000/api/rules')
    print("\nTesting GET /api/rules:")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of rules: {data['rules_count']}")
    print("\nFirst rule example:")
    if data['rules']:
        print(json.dumps(data['rules'][0], indent=2))

def test_get_rules_with_params():
    # Test getting rules with custom parameters
    params = {
        'min_support': 0.02,
        'min_confidence': 0.4
    }
    response = requests.get('http://localhost:5000/api/rules', params=params)
    print("\nTesting GET /api/rules with custom parameters:")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of rules: {data['rules_count']}")

def test_download_rules():
    # Test downloading rules file
    response = requests.get('http://localhost:5000/api/download/rules')
    print("\nTesting GET /api/download/rules:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        with open('downloaded_rules.json', 'wb') as f:
            f.write(response.content)
        print("Rules file downloaded successfully")

if __name__ == '__main__':
    test_get_rules()
    test_get_rules_with_params()
    test_download_rules()
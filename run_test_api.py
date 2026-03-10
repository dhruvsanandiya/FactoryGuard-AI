"""Simple script to test the FactoryGuard AI API"""

import requests
import json

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get('http://127.0.0.1:5000/api/v1/health')
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is the API running?")
        print("Start it with: python run_api.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_predict():
    """Test prediction endpoint"""
    print("\nTesting prediction endpoint...")
    try:
        data = {
            "machine_id": "M_204",
            "temperature": 82.4,
            "pressure": 1.9,
            "vibration": 0.02
        }
        response = requests.post(
            'http://127.0.0.1:5000/api/v1/predict',
            json=data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is the API running?")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get('http://127.0.0.1:5000/api/v1/model/info')
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is the API running?")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("FactoryGuard AI API Test")
    print("=" * 60)
    
    # Test all endpoints
    health_ok = test_health()
    if health_ok:
        test_predict()
        test_model_info()
    
    print("\n" + "=" * 60)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app import app
from fastapi.testclient import TestClient
import pytest
import json
import datetime

'''
    Following test cases are checked:

    1. If the index page with port 8212 is accessible or not.
    2. Check if the post request is successful in ../qp_similarity
    3. Check if the overall server latency is less than 200ms
'''

class TestTransaction:
    def setup_method(self):
        self.start_time = datetime.datetime.now()
        self.api_client = TestClient(app)
        self.sample_json = json.load(open("inputs/sample.json", "rb"))     

    def test_index(self, monkeypatch: pytest.MonkeyPatch):
        response = self.api_client.get('/')
        assert response.status_code == 200

    def test_transaction_latency(self, monkeypatch: pytest.MonkeyPatch):
        response = self.api_client.post('/qp_similarity', 
                                        json=self.sample_json)
        assert response.status_code == 200
        self.end_time = datetime.datetime.now()
        latency = (self.end_time - self.start_time).total_seconds() * 1000
        assert latency < 200 # check if overall server latency is less than 200ms

    def teardown_method(self):
        pass

# Por ejemplo en backend/recommender/tests/test_dummy.py
from django.test import TestCase


class TestDummy(TestCase):
    def test_truth(self):
        self.assertTrue(True)

from django.test import TestCase

from borough.models import Borough


class BoroughModelTest(TestCase):
    def test_all_expected_boroughs_are_present(self):
        expected_boroughs = {
            "Barking and Dagenham",
            "Barnet",
            "Bexley",
            "Brent",
            "Bromley",
            "Camden",
            "City of London",
            "Croydon",
            "Ealing",
            "Enfield",
            "Greenwich",
            "Hackney",
            "Hammersmith and Fulham",
            "Haringey",
            "Harrow",
            "Havering",
            "Hillingdon",
            "Hounslow",
            "Islington",
            "Kensington and Chelsea",
            "Kingston upon Thames",
            "Lambeth",
            "Lewisham",
            "Merton",
            "Newham",
            "Redbridge",
            "Richmond upon Thames",
            "Southwark",
            "Sutton",
            "Tower Hamlets",
            "Waltham Forest",
            "Wandsworth",
            "Westminster",
        }

        db_boroughs = set(Borough.objects.values_list("name", flat=True))

        self.assertEqual(
            db_boroughs,
            expected_boroughs,
            f"Mismatch in borough names.\nMissing: {expected_boroughs - db_boroughs}\nUnexpected: {db_boroughs - expected_boroughs}",
        )

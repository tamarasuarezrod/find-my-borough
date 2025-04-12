import pandas as pd

def load_zone_data():
    borough_zone_mapping = {
        "barking and dagenham": 4,
        "barnet": 4,
        "bexley": 4,
        "brent": 3,
        "bromley": 4,
        "camden": 1,
        "city of london": 1,
        "croydon": 4,
        "ealing": 3,
        "enfield": 4,
        "greenwich": 3,
        "hackney": 2,
        "hammersmith and fulham": 2,
        "haringey": 3,
        "harrow": 4,
        "havering": 4,
        "hillingdon": 4,
        "hounslow": 4,
        "islington": 2,
        "kensington and chelsea": 1,
        "kingston upon thames": 4,
        "lambeth": 2,
        "lewisham": 3,
        "merton": 4,
        "newham": 3,
        "redbridge": 4,
        "richmond upon thames": 4,
        "southwark": 2,
        "sutton": 4,
        "tower hamlets": 2,
        "waltham forest": 3,
        "wandsworth": 2,
        "westminster": 1,
    }

    df_zones = pd.DataFrame(list(borough_zone_mapping.items()), columns=["borough", "zone"])

    df_zones.to_csv("data/clean/borough_zones.csv", index=False)
    
    print("-- Zone data cleaned and saved to data/clean/borough_zones.csv")

    return df_zones

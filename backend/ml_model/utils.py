valid_boroughs = [
    'barking and dagenham', 'barnet', 'bexley', 'brent', 'bromley', 'camden', 'city of london',
    'croydon', 'ealing', 'enfield', 'greenwich', 'hackney', 'hammersmith and fulham',
    'haringey', 'harrow', 'havering', 'hillingdon', 'hounslow', 'islington',
    'kensington and chelsea', 'kingston upon thames', 'lambeth', 'lewisham', 'merton',
    'newham', 'redbridge', 'richmond upon thames', 'southwark', 'sutton',
    'tower hamlets', 'waltham forest', 'wandsworth', 'westminster'
]

def standardize_borough_names(series):
    replacements = {
        'westminster city': 'westminster'
    }

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace('&', 'and')
        .replace(replacements)
    )


def filter_valid_boroughs(df, column, valid):
    return df[df[column].isin(valid)].copy()

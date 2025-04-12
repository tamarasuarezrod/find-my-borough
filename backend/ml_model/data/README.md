# Data Sources for FindMyBorough

This folder contains datasets used to build and train the recommendation model for FindMyBorough.

## 1. Rental Prices by Borough

- Name: Private Rental Market Summary Statistics
- Source: London Datastore
- Direct Link: https://data.london.gov.uk/dataset/average-private-rents-borough
- File Used: voa-average-rent-borough.xls
- Date Accessed: April 2025

This dataset includes median monthly private rental prices for various property types across London boroughs.

## 2. Crime by Borough

- Name: MPS Monthly Crime Dashboard Data
- Source: Metropolitan Police Service via London Datastore
- Direct Link: https://data.london.gov.uk/dataset/mps-monthly-crime-dahboard-data
- File Used: MPS Monthly Crime Dashboard_TNOCrimeData.csv
- Date Range: From 01/03/2021 to 28/02/2025
- Date Accessed: April 2025

This dataset includes Total Notifiable Offences (TNO) by borough across the specified time period.

## 3. Population by Age and Borough

- Name: London Borough Population by Age
- Source: Office for National Statistics (ONS)
- Direct Link: [ONS FOI Request](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/londonboroughpopulationbyageethnicityandhouseholdtype)
- File Used: londonboroughs.xlsx (Sheet: Age London Boroughs)
- Date Accessed: April 2025

This dataset provides population counts by detailed age categories across London boroughs. It was used to estimate the proportion of young adults (aged 20–34) in each borough to inform the youthful population feature.

## 4. Borough Zones (Centrality)

- Name: Borough to Zone Mapping (Custom)
- Source: Manually curated from Transport for London zone references and general borough geography.
- File Used: borough_zones.csv
- Date Accessed: April 2025

This is a custom dataset mapping each borough to an approximate travel zone (1 to 4), representing proximity to central London. Lower values indicate higher centrality.

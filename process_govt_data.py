"""Process downloaded data.gov.in JSON file."""
import json
import pandas as pd
import os

# Load the downloaded file
with open('data/gov_crop_data.json', 'r') as f:
    data = json.load(f)

print('=' * 60)
print('OFFICIAL GOVERNMENT DATA LOADED')
print('=' * 60)
print(f"Source: {data['source']}")
print(f"Title: {data['title']}")
print(f"Total Records Available: {data['total']:,}")
print(f"Records Downloaded: {data['count']}")
print(f"Organizations: {data['org']}")
print()

# Extract records
records = data['records']
df = pd.DataFrame(records)

print(f"Sample Data ({len(df)} records):")
print(df.to_string())

print()
print('=' * 60)
print('TO GET HARYANA/KARNAL RICE DATA:')
print('=' * 60)
print()
print('Open this URL in your browser:')
print()
print('https://api.data.gov.in/resource/35be999b-0208-4354-b557-f6ca9a5355de?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&limit=10000&filters[state_name]=Haryana')
print()
print('Save as: d:/tr/pmfby_engine/data/haryana_data.json')

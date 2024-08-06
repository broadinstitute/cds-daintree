import pandas as pd
import argparse
import requests


def verify_feature_info(feature_info_df, json_url):
    response = requests.get(json_url, headers={'accept': 'application/json'})
    if response.status_code == 200:
        json_data = response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    json_df = pd.DataFrame(json_data)

    json_df['id'] = json_df['id'].astype(str)

    merged_df = feature_info_df.merge(json_df, left_on=['feature_label', 'given_id'], right_on=['label', 'id'], how='left')

    mismatches = merged_df[merged_df['id'].isnull() | merged_df['label'].isnull()]

    # Print mismatches
    print("Mismatches found:")
    print(mismatches)

    if mismatches.empty:
        print("No mismatches found.")
    else:
        mismatches.to_csv('mismatches.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Verify feature information against a JSON file.')
    parser.add_argument('feature_info_path', type=str, help='Path to the feature_info CSV file.')
    parser.add_argument('json_url', type=str, help='URL to fetch the JSON data from.')

    args = parser.parse_args()

    feature_info_df = pd.read_csv(args.feature_info_path)

    verify_feature_info(feature_info_df, args.json_url)

if __name__ == "__main__":
    main()

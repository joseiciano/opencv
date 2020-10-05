# USAGE
# python search_bing_api.py --query "charmander" --output dataset/charmander
# python search_bing_api.py --query "pikachu" --output dataset/pikachu
# python search_bing_api.py --query "squirtle" --output dataset/squirtle
# python search_bing_api.py --query "bulbasaur" --output dataset/bulbasaur
# python search_bing_api.py --query "mewtwo" --output dataset/mewtwo

# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

# API Key and query parameters
API_KEY = "28ce8b22fcbe4f3188e298f8e72b45de"
MAX_RESULTS = 10
GROUP_SIZE = 50

# API endpoint
#URL = "https://api.cognitive.microsoft.com/bing/v5.0/images/search"
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
# Set of extensions we can encounter from API
EXCEPTIONS = set([IOError, FileNotFoundError,
                  exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])

# Store search term then set headers/search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# Search
print(f"[INFO] searching Bing API for '{term}'")
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# Grab results from search, including number of estimated results
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print(f"[INFO] {estNumResults} total results for '{term}'")

# Initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
    # Update search parameter by current offset
    print(
        f"[INFO] making request for group {offset}-{offset+GROUP_SIZE} of {estNumResults}...")
    params["offset"] = offset

    # Fetch results given current offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print(
        f"[INFO] saving images for group {offset}-{offset+GROUP_SIZE} of {estNumResults}...")

    # Loop over the results, trying to download it
    for v in results["value"]:
        try:
            # Download image request
            print(f"[INFO] fetching: {v['contentUrl']}")
            r = requests.get(v["contentUrl"], timeout=30)

            # Build path to output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)])

            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # Catch errors that would not allow us to download image
        except Exception as e:
            if type(e) in EXCEPTIONS:
                print(f"[INFO] skipping: {v['contentUrl']}")
                continue

        # Try to load the image from disk
        image = cv2.imread(p)

        # If image is none, ignore it as could not load from disk
        if image is None:
            print(f"[INFO] deleting: {p}")
            os.remove(p)
            continue

        # update the counter
        total += 1

from requests import exceptions
import argparse
import requests
import cv2
import os

# Microsft Cognitive Services API Info
API_KEY = ""
MAX_RESULTS = 250  # Max number of results for given search
GROUP_SIZE = 50  # Group size for results (max of 50 per request)

# Endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# Exceptions possible from requests library
EXCEPTIONS = set([IOError, FileNotFoundError,
                  exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])

# Search term
# term = args["query"]
term = "Jeff Goldblum young"
output = f"dataset/test"

# API Stuff, required
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

# Total number of images successfully downloaded
total = 0

# Loop over the estimated number of results in `GROUP_SIZE` groups
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
            p = os.path.sep.join([output, f"{str(total).zfill(8)}{ext}"])

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

        # Update the counter
        total += 1

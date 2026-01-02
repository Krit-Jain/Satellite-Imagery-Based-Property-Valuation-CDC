import os
import time
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"

def fetch_image(lat, lon, zoom=18, size=256):
    url = (
        f"{BASE_URL}/"
        f"{lon},{lat},{zoom}/"
        f"{size}x{size}"
        f"?access_token={MAPBOX_TOKEN}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def download_images(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(output_dir, f"{row['id']}.png")

        if os.path.exists(img_path):
            continue

        try:
            img = fetch_image(row["lat"], row["long"])
            img.save(img_path)
            time.sleep(0.05)  # rate limiting
        except Exception as e:
            print(f"Failed for ID {row['id']}: {e}")

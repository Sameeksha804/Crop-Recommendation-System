import requests
import os

def download_dataset():
    # URL of the dataset
    url = "https://raw.githubusercontent.com/patel-zeel/crop-recommendation-system/master/Crop_recommendation.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    print("Downloading crop recommendation dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save the dataset
        with open('Crop_recommendation.csv', 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully!")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        print("Please download the dataset manually from:")
        print("https://www.kaggle.com/datasets/patelris/crop-recommendation-dataset")
        print("and place it in the project root directory as 'Crop_recommendation.csv'")

if __name__ == "__main__":
    download_dataset() 
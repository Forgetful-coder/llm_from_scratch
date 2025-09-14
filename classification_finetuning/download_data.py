import urllib.request as request
import ssl
import os
import zipfile
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "data/classification_data/sms_spam_collection.zip"
extracted_path = "data/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"




def download_and_unzip(url,zip_path,extracted_file_path,data_file_path):

    if data_file_path.exists():
        print(f"The {data_file_path} exists hence skipping download and unzipping 😎")

    ssl_context = ssl._create_unverified_context()

    os.makedirs(os.path.dirname(zip_path),exist_ok=True)
    #Dowloading the file:
    with request.urlopen(url,context=ssl_context) as response:
       
        with open(zip_path,'wb') as out_file:
            out_file.write(response.read())


    #Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"Original file path after rename {original_file_path}")
    print(f"File downloaded and saved as {data_file_path}")


if __name__== "__main__":
    download_and_unzip(url,zip_path,extracted_path,data_file_path)


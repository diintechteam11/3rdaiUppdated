import os
import boto3
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")

def check_r2():
    print("--- R2 STORAGE CHECK ---")
    if not all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET]):
        print("Error: Missing R2 environmental variables.")
        return

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name='auto'
        )
        
        print(f"Connecting to bucket: {R2_BUCKET}")
        response = s3.list_objects_v2(Bucket=R2_BUCKET, MaxKeys=10, Prefix="recordings/")
        
        if 'Contents' in response:
            print(f"\nFound {len(response['Contents'])} recent recordings in R2:")
            for obj in response['Contents']:
                print(f" - {obj['Key']} ({obj['Size'] / 1024 / 1024:.2f} MB)")
        else:
            print("\nNo recordings found in R2 yet.")
            
    except Exception as e:
        print(f"Error connecting to R2: {e}")

if __name__ == "__main__":
    check_r2()

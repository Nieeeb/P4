from minio import Minio
# file_uploader.py MinIO Python SDK example
from minio.error import S3Error
import glob
import os
from tqdm import tqdm

def upload_local_directory_to_minio(local_path, bucket_name, minio_path, client):
        assert os.path.isdir(local_path)

        for local_file in tqdm(glob.glob(local_path + '/**')):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file), client)
            else:
                remote_path = os.path.join(
                    minio_path, local_file)
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                split_remote = remote_path.split('/v1/')[1]
                #print(split_remote)
                #print(f"Uploading {local_file} to {split_remote}", flush=True)
                client.fput_object(bucket_name, split_remote, local_file)

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("s3.schniebster.dk",
        access_key="wP4",
        secret_key="harborfront",
    )

    path = r"/home/nieb/Projects/Big Data/Images/Seasons_drift/v1/harborfrontv1/"

    upload_local_directory_to_minio(local_path=path, bucket_name="thermaldrift", minio_path='', client=client)

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
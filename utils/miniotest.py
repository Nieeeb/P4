from minio import Minio
# file_uploader.py MinIO Python SDK example
from minio.error import S3Error

def main():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("s3.schniebster.dk",
        access_key="wP4",
        secret_key="harborfront",
    )

    # The file to upload, change this path if needed
    source_file = "Lecture1/test.py"

    # The destination bucket and filename on the MinIO server
    bucket_name = "thermaldrift"
    destination_file = "test.py"

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if found:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)

import yaml
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def compute_eigenvalues(img_path: str, noisy_img: bool):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if noisy_img:
        noise_sigma = 50
        img = img.astype(np.float32) + np.random.normal(0, noise_sigma, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

    pca = PCA()
    pca.fit(img)
    
    eigenvalues = pca.explained_variance_
    return eigenvalues

# Load image (replace 'my_image.jpg' with your image file)
img = r"C:\Users\Victor Steinrud\Downloads\ed7a-how-to-buy-a-puppy-article-dog.png"
if img is None:
    raise ValueError("Image not found or unable to load.")


def noise_estimaiton(file_txt, img_folder) -> list:
    filenames = []
            
    with open(file_txt) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(img_folder + filename)

    eigenvalues = []

    for file in filenames:
        eigenvalues.append(compute_eigenvalues(file, noisy_img=False))

    return np.mean(eigenvalues, axis=0)






def main():

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    with open(args.args_file) as cf_file:
        params = yaml.safe_load( cf_file.read())

    mean_eigenvalues = noise_estimaiton(file_txt=params.get('train_txt'), img_folder=params.get('train_imgs'))

    


if __name__ == "__main__":


# # print(len(eigenvalues_original))
# # eigenvalues_noisy = compute_eigenvalues(patches_noisy)

# # Plot the original and noisy images side by side.
# plt.figure(figsize=(12, 4))

# # plt.subplot(1, 3, 1)
# # plt.imshow(img, cmap='gray')
# # plt.title('Original Image')
# # plt.axis('off')

# # plt.subplot(1, 3, 2)
# # plt.imshow(noisy_img, cmap='gray')
# # plt.title('Noisy Image')
# # plt.axis('off')

# # Plot the eigenvalue spectra.
# # plt.subplot(1, 3, 3)
# plt.plot(eigenvalues_original, marker='o', linestyle='-', label='Original')
# # plt.plot(eigenvalues_noisy, marker='o', linestyle='-', label='Noisy')
# plt.yscale('log')
# plt.title('Eigenvalue Spectrum')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalues')
# plt.legend()

# plt.tight_layout()
# plt.show()


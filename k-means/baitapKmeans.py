# Import các thư viện cần thiết
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import os

# Hàm kiểm tra sự tồn tại của ảnh
def check_image_files(image_files):
    valid_image_files = [file for file in image_files if os.path.exists(file)]
    if not valid_image_files:
        print("Không có file ảnh hợp lệ. Vui lòng kiểm tra đường dẫn ảnh.")
    return valid_image_files

# Hàm đọc ảnh và phân cụm K-means
def perform_kmeans(image_rgb, n_clusters=2):
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    labels_kmeans = kmeans.labels_
    segmented_image_kmeans = labels_kmeans.reshape(image_rgb.shape[:2])
    return segmented_image_kmeans

# Hàm đọc ảnh và phân cụm Fuzzy C-means
def perform_fcm(image_rgb, n_clusters=2):
    pixels = image_rgb.reshape((-1, 3))
    pixels_fcm = pixels.T  # Chuyển vị để phù hợp với FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(pixels_fcm, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    labels_fcm = np.argmax(u, axis=0)
    segmented_image_fcm = labels_fcm.reshape(image_rgb.shape[:2])
    return segmented_image_fcm

# Hàm hiển thị ảnh gốc, K-means và Fuzzy C-means
def display_results(image_rgb, segmented_image_kmeans, segmented_image_fcm, idx, axes):
    # Hiển thị ảnh gốc
    axes[idx, 0].imshow(image_rgb)
    axes[idx, 0].set_title(f"Ảnh gốc {idx + 1}")
    axes[idx, 0].axis('off')

    # Hiển thị kết quả phân cụm K-means
    axes[idx, 1].imshow(segmented_image_kmeans, cmap='viridis')
    axes[idx, 1].set_title(f"Phân cụm K-means {idx + 1}")
    axes[idx, 1].axis('off')

    # Hiển thị kết quả phân cụm FCM
    axes[idx, 2].imshow(segmented_image_fcm, cmap='viridis')
    axes[idx, 2].set_title(f"Phân cụm Fuzzy C-means {idx + 1}")
    axes[idx, 2].axis('off')

# Hàm chính để thực hiện các bước phân cụm và hiển thị kết quả
def main(image_files, n_clusters=2):
    # Kiểm tra các file ảnh
    valid_image_files = check_image_files(image_files)

    # Nếu có ảnh hợp lệ, tiếp tục xử lý
    if valid_image_files:
        # Khởi tạo figure với lưới 3 hàng và số ảnh x 3 cột (gốc, K-means, FCM)
        fig, axes = plt.subplots(len(valid_image_files), 3, figsize=(15, 5 * len(valid_image_files)))

        # Lặp qua từng ảnh để phân cụm
        for idx, file in enumerate(valid_image_files):
            # Đọc ảnh vệ tinh
            image = cv2.imread(file)
            if image is None:
                print(f"Không thể đọc file: {file}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Thực hiện phân cụm K-means
            segmented_image_kmeans = perform_kmeans(image_rgb, n_clusters)

            # Thực hiện phân cụm Fuzzy C-means
            segmented_image_fcm = perform_fcm(image_rgb, n_clusters)

            # Hiển thị kết quả
            display_results(image_rgb, segmented_image_kmeans, segmented_image_fcm, idx, axes)

        # Tăng khoảng cách giữa các hàng để dễ nhìn hơn
        plt.tight_layout()
        plt.show()

# Đường dẫn danh sách các file ảnh
image_files = [
    'C:/Users/admin/Downloads/XLABai1/k-means/dauvao/anhvetinh.jpg',
    'C:/Users/admin/Downloads/XLABai1/k-means/dauvao/anhvetinh2.jpg'
]

# Gọi hàm main để thực hiện phân cụm và hiển thị
main(image_files, n_clusters=2)

from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

# 替换为你的文件路径
# 1M
# original_mat_path = 'FF.mat'
# reconstructed_mat_path = 'F_recon_1M/accu50.mat'
# # 加载数据
# original_data = loadmat(original_mat_path)
# reconstructed_data = loadmat(reconstructed_mat_path)
# original_image = original_data['FF']
# reconstructed_image = reconstructed_data['accumulatedData_cuda']


# 10M
original_mat_path = 'FF_10M.txt'
reconstructed_mat_path = 'F_10M_p50.txt'
RES = 20940
original_data = np.loadtxt(original_mat_path)
original_image = original_data.reshape(RES, RES)
reconstructed_data = np.loadtxt(reconstructed_mat_path)
reconstructed_image = reconstructed_data.reshape(RES, RES)


# 计算PSNR
# def PSNR(original_image, reconstructed_image):
#     psnr_value = psnr(original_image, reconstructed_image, data_range=reconstructed_image.max() - reconstructed_image.min())
#     print(f"PSNR: {psnr_value}")

def PSNR(original, reconstructed, max_pixel=255.0):
    # 计算均方误差 (MSE)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # 如果图像完全相同，PSNR 为无穷大
    # 计算 PSNR
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    print(f"PSNR: {psnr_value}")

# 计算SSIM
def SSIM(original_image, reconstructed_image):
    ssim_value, _ = ssim(original_image, reconstructed_image, full=True, data_range=reconstructed_image.max() - reconstructed_image.min())
    print(f"SSIM: {ssim_value}")

# 计算MSE
def MSE(original_image, reconstructed_image):
    mse_value = mse(original_image, reconstructed_image)
    print(f"MSE: {mse_value}")



# # 线性缩放
def linear_scaling(image, target_min=0, target_max=1, axis=None):
    """
    线性缩放图像的像素值。
    axis 参数用于指定按行(0)或列(1)进行缩放；如果为 None，则对整个图像进行缩放。
    """
    if axis is not None:
        scaled_image = np.empty_like(image, dtype=np.float64)
        if axis == 0:  # 按行
            for i in range(image.shape[0]):
                min_val = image[i, :].min()
                max_val = image[i, :].max()
                scaled_image[i, :] = (image[i, :] - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
        elif axis == 1:  # 按列
            for i in range(image.shape[1]):
                min_val = image[:, i].min()
                max_val = image[:, i].max()
                scaled_image[:, i] = (image[:, i] - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
        return scaled_image
    else:
        min_val = image.min()
        max_val = image.max()
        print(min_val, max_val)
        return (image - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

# # 对数变化
def apply_signed_log_penalty(image, epsilon=1e-10):
    transformed_image = np.zeros_like(image, dtype=float)
    # 找到图像中所有大于0的像素位置
    positive_mask = image > 1
    # 仅对这些位置的像素应用对数变换
    transformed_image[positive_mask] = np.log(image[positive_mask])
    return transformed_image

def apply_signed_log_penalty2(image, epsilon=1e-10):
    signs = np.sign(image)
    # 对绝对值应用对数（添加一个小的常数epsilon以避免对0取对数）
    result = signs * np.log(np.abs(image) + epsilon)
    # result = signs * np.log(np.abs(image))
    return result




if __name__ == '__main__':

    # 线性变换(全部)
    # original_image = linear_scaling(image=original_image)
    # reconstructed_image = linear_scaling(image=reconstructed_image)

    # 对数变换abs
    # original_image = apply_signed_log_penalty2(image=original_image)
    # reconstructed_image = apply_signed_log_penalty2(image=reconstructed_image)

    # 对数变换gt0
    original_image = apply_signed_log_penalty(image=original_image)
    reconstructed_image = apply_signed_log_penalty(image=reconstructed_image)

    PSNR(original_image, reconstructed_image)
    SSIM(original_image, reconstructed_image)
    MSE(original_image, reconstructed_image)
   
        

        
import os
import rasterio
from rasterio.plot import reshape_as_image
from itertools import combinations
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 定义文件夹路径
base_folder = r'H:\projects\21-24'
pairs = [
    ('preFireS1', 'preFireS2', 'preFireS2_indices'),
    ('postFireS1', 'postFireS2', 'postFireS2_indices'),
]

# 定义波段名称
band_names = {
    'S1': ['VV', 'VH'],
    'S2': ["B", "G", "R", "RE1", "RE2", "RE3", "NIR", "RE4", "S1", "S2"],
    'indices': ['NDVI', 'EVI', 'BSI', 'IBI', 'SSI', 'RVI', 'DVI', 'GCVI', 'NBR', 'NBR2', 'BAI', 'MIRBI', 'NBR_PLUS',
                'BAIS2', 'REP']
}

# 定义需要组合的波段
# 定义三组波段
group1 = ["RE4", "S2", "S1", "RE1"]  # 第一组
group2 = ["MIRBI", "NBR2", "NBR_PLUS"]  # 第二组
group3 = ["VV", "VH"]  # 第三组

# 存储所有组合的列表
all_combinations = []

# 情况 1：第三组选择 2 个（VV 和 VH）
if len(group3) == 2:
    for g2_count in range(0, 2):  # 第二组选择 0 到 1 个（因为第二组最多选 2 个，但第一组至少选 3 个）
        for g2_selected in combinations(group2, g2_count):
            g1_needed = 6 - len(g2_selected) - 2  # 第一组需要选的波段数
            if 3 <= g1_needed <= len(group1):  # 确保第一组最少选 3 个
                for g1_selected in combinations(group1, g1_needed):
                    combination = list(g2_selected) + group3 + list(g1_selected)
                    all_combinations.append(combination)

# 情况 2：第三组选择 0 个
for g2_count in range(0, 3):  # 第二组选择 0 到 2 个
    if g2_count <= 2:  # 确保第二组最多选 2 个
        for g2_selected in combinations(group2, g2_count):
            g1_needed = 6 - len(g2_selected)  # 第一组需要选的波段数
            if 3 <= g1_needed <= len(group1):  # 确保第一组最少选 3 个
                for g1_selected in combinations(group1, g1_needed):
                    combination = list(g2_selected) + list(g1_selected)
                    all_combinations.append(combination)

# 生成所有可能的 6 波段组合
band_combinations = all_combinations

# 创建输出文件夹
output_base_folder = os.path.join(base_folder, 'output')
os.makedirs(output_base_folder, exist_ok=True)


def normalize_and_convert(data):
    """逐波段归一化数据到0-255并转换为uint8类型"""
    # 检查数据是否为多波段（三维数组）
    if len(data.shape) == 3:
        normalized_data = np.zeros_like(data, dtype=np.uint8)
        for band_idx in range(data.shape[0]):  # 遍历每个波段
            band_data = data[band_idx, :, :]
            band_min, band_max = np.min(band_data), np.max(band_data)
            if band_max - band_min == 0:  # 防止除以零
                normalized_data[band_idx, :, :] = band_data
            else:
                normalized_data[band_idx, :, :] = ((band_data - band_min) / (band_max - band_min)) * 255
        return normalized_data.astype(np.uint8)
    else:
        # 单波段数据
        band_min, band_max = np.min(data), np.max(data)
        if band_max - band_min == 0:  # 防止除以零
            return data
        else:
            normalized_data = ((data - band_min) / (band_max - band_min)) * 255
            return normalized_data.astype(np.uint8)


def process_single_file(args):
    """处理单个文件的波段组合并生成图像"""
    file_name, pair, combo = args
    try:
        # 文件路径
        file_path_s1 = os.path.join(base_folder, pair[0], file_name)
        file_path_s2 = os.path.join(base_folder, pair[1], file_name)
        file_path_indices = os.path.join(base_folder, pair[2], file_name)

        # 读取 S1 和 S2 文件
        with rasterio.open(file_path_s1) as src_s1, rasterio.open(file_path_s2) as src_s2, rasterio.open(
                file_path_indices) as src_indices:
            # 提取 S1 和 S2 波段
            s1_bands = band_names['S1']
            s2_bands = band_names['S2']
            indices_bands = band_names['indices']

            s1_data = {s1_bands[i]: src_s1.read(i + 1) for i in range(src_s1.count)}
            s2_data = {s2_bands[i]: src_s2.read(i + 1) for i in range(src_s2.count)}
            indices_data = {indices_bands[i]: src_indices.read(i + 1) for i in range(src_indices.count)}

            # 合并 S1、S2 和植被指数数据
            combined_data = {**s1_data, **s2_data, **indices_data}

            # 检查波段是否存在
            if not all(band in combined_data for band in combo):
                print(f"Missing bands in {file_name}: {combo}")
                return

            # 提取波段数据并归一化
            selected_data = [normalize_and_convert(combined_data[band]) for band in combo]

            # 获取元数据
            meta = src_s1.meta
            meta.update({
                "count": len(combo),  # 波段数量
                "dtype": np.uint8,  # 数据类型
                "driver": "GTiff"  # 输出格式
            })

            # 输出文件夹和路径
            combo_folder = os.path.join(output_base_folder, '_'.join(combo))
            output_sub_folder = os.path.join(combo_folder, 'A' if 'pre' in pair[0] else 'B')
            os.makedirs(output_sub_folder, exist_ok=True)

            output_file = os.path.join(output_sub_folder, file_name)

            # 写入 GeoTIFF 文件
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_data in enumerate(selected_data, start=1):
                    dst.write(band_data, i)  # 写入波段数据
                    dst.set_band_description(i, combo[i - 1])  # 设置波段描述

    except Exception as e:
        print(f"Error processing {file_name} with combo {combo}: {e}")


def process_band_combinations():
    """处理所有文件和波段组合"""
    # 遍历所有波段组合
    for combo in tqdm(band_combinations, desc="Processing band combinations"):
        # 遍历所有文件
        file_names = os.listdir(os.path.join(base_folder, pairs[0][0]))  # 获取基准文件列表
        args_list = [(file_name, pair, combo) for file_name in file_names for pair in pairs]

        # 使用多进程处理
        with Pool(cpu_count()//2) as pool:
            list(tqdm(pool.imap(process_single_file, args_list), total=len(args_list), desc=f"Processing combo {combo}", leave=False))


if __name__ == "__main__":
    process_band_combinations()
    print("Processing complete.")
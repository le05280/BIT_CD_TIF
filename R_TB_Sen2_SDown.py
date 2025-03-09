import ee
import geemap
from datetime import datetime
import logging
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
import os
import glob
import requests
import shutil
import pandas as pd
from retry import retry

print('start')
ee.Initialize()

print('end')
# Create an empty list to store longitude, latitude, and time information
coordinates_list = []
# Used to store combinations of dates and points that have already been processed to ensure uniqueness
processed_combos = []
# Used to store the names of images that have already been exported to prevent duplicate exports
exported_images = []
# Used to store processing results
results_list = []

# Read the CSV file E:\Projects\GEE\Cn-all
csv_file_path = 'S2center_points.csv'
df = pd.read_csv(csv_file_path)

# Set the folder path   D:\Projects\PycharmProjects\GEE\China-Land8-all
folder_path = "./Pre"

params = {
    'count': 100,  # How many image chips to export
    'buffer': 6400,  # The buffer distance (m) around each point
    'scale': 20,  # The scale to do stratified sampling
    'seed': 1,  # A randomization seed to use for subsampling.
    'dimensions': '640x640',  # The dimension of each image chip
    'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': 'tile_',  # The filename prefix
    'processes': 7,  # How many processes to used for parallel processing
    'Pre': './Pre',  # The label output directory. Default to the current working directly
    'Post': './Post',  # The val output directory. Default to the current working directly
}
# Get all file paths
file_paths = glob.glob(os.path.join(folder_path, "*"))

# Get all file names (excluding extensions)
file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

# Remove the last underscore and everything after it
file_names_without_suffix = [name[:name.rfind('_')] if '_' in name else name for name in file_names]

# Write the existing files in the folder to the list
exported_images = file_names_without_suffix


# 去云函数，适用于 Sentinel-2
def maskS2sr(image):
    qa = image.select('QA60')  # 使用 'QA60' 波段
    cloudBitMask = (1 << 10)  # 云位
    cirrusBitMask = (1 << 11)  # 卷云位
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)



# 使用google的云得分影像集合 S2CS
S2CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')


# 定义MergeImages函数
def MergeImages(primary, secondary):
    join = ee.Join.inner()
    filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    Col = join.apply(primary, secondary, filter).map(lambda image: ee.Image.cat(
        ee.Image(image.get("primary")),
        ee.Image(image.get("secondary"))
    ))
    return ee.ImageCollection(Col)

# 定义一个函数来更新影像的掩膜
def update_mask(img):
    cloud_score = img.select('cs')
    return img.updateMask(cloud_score.gte(0.6))

# 去云函数
def maskL8sr(image):
    qa = image.select('QA_PIXEL')  # 使用 'QA_PIXEL' 波段
    cloudShadowBitMask = (1 << 3)  # 云影位
    cloudsBitMask = (1 << 5)  # 云位
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


def loaddata(date_start, date_end, roi):
    s2SR_Clo = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    s2SR_Post2 = s2SR_Clo.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
                                 ["B", "G", "R", "RE1", "RE2", "RE3", "NIR", "RE4", "S1", "S2"]).sort(
        'CLOUDY_PIXEL_PERCENTAGE')
    #     使用谷歌去云算法!
    s2SR_Post2 = MergeImages(s2SR_Post2, S2CS).map(update_mask).median().clip(roi)  # 中值合成

    # 获取去云后的 Landsat 8 影像
    l8SR_Post = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(roi)
            .filterDate(date_start, date_end)
            .map(maskL8sr)  # 应用去云函数
            .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'],
                    ["B", "G", "R", "NIR", "S1", "S2", "T1"])
            .sort('CLOUD_COVER', True)
            .median()  # 中值合成
            .clip(roi)
    )

    # 创建 Sentinel-1 GRD 影像集合
    S1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.eq('resolution_meters', 50))
          .filterDate(date_start, date_end)
          .filterBounds(roi)
          .select(['VV', 'VH'])
          .median()  # 中值合成
          .clip(roi))




    return s2SR_Post2, S1, l8SR_Post


'''
LANDSAT/LC08/C02/T1_L2
This dataset contains atmospherically corrected surface reflectance and surface temperature derived from Landsat 8 OLI/TIRS sensor data.
These images include 5 visible and near-infrared (VNIR) bands and 2 shortwave infrared (SWIR) bands processed with orthorectified surface reflectance,
as well as 1 thermal infrared (TIR) band processed with orthorectified surface temperature. They also contain intermediate bands for ST product calculation and QA bands.
Band names are: SR_B*
LANDSAT/LC08/C02/T1:
Landsat scenes with the highest available data quality are placed in Tier 1 and are considered suitable for time series processing analysis. Tier 1 includes Level 1 Precision Terrain (L1TP) processed data,
with good feature radiometry and cross-calibration across different Landsat sensors. Tier 1 scenes will have consistent geographic registration
and within the specified tolerance [<= 12 m Root Mean Square Error (RMSE)]. All Tier 1 Landsat data can be considered consistent and cross-calibrated (regardless of sensor) throughout the collection. See USGS documentation for more information.
'''


# 计算两个影像都有的波段的指数
def common_indices(img):
    NDVI = img.normalizedDifference(['NIR', 'R']).rename('NDVI')
    EVI = img.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)', {
            'RED': img.select('R'),  # 620-670nm, RED
            'NIR': img.select('NIR'),  # 841-876nm, NIR
            'BLUE': img.select('B')  # 459-479nm, BLUE
        }).rename("EVI")

    BSI = img.expression('((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE))', {
        'RED': img.select('R'),
        'BLUE': img.select('B'),
        'NIR': img.select('NIR'),
        'SWIR1': img.select('S1'),
    }).rename('BSI')

    IBI = img.expression(
        '(2 * SWIR1 / (SWIR1 + NIR) - (NIR / (NIR + RED) + GREEN / (GREEN + SWIR1))) / (2 * SWIR1 / (SWIR1 + NIR) + (NIR / (NIR + RED) + GREEN / (GREEN + SWIR1)))',
        {
            'SWIR1': img.select('S1'),
            'NIR': img.select('NIR'),
            'RED': img.select('R'),
            'GREEN': img.select('G')
        }).rename('IBI')

    SSI = img.expression('(NIR + RED + GREEN + BLUE)', {
        'BLUE': img.select('B'),
        'NIR': img.select('NIR'),
        'RED': img.select('R'),
        'GREEN': img.select('G')
    }).rename('SSI')

    RVI = img.expression('NIR / RED', {
        'NIR': img.select('NIR'),
        'RED': img.select('R')
    }).rename("RVI")

    DVI = img.expression('NIR - RED', {
        'NIR': img.select('NIR'),
        'RED': img.select('R')
    }).rename('DVI')

    GCVI = img.expression('NIR / GREEN - 1', {
        'NIR': img.select('NIR'),
        'GREEN': img.select('G')
    }).rename('GCVI')
    NBR = img.expression('(NIR - SWIR2) / (NIR + SWIR2)', {
        'NIR': img.select('NIR'),
        'SWIR2': img.select('S2')
    }).rename('NBR')

    NBR2 = img.expression('(SWIR1 - SWIR2) / (SWIR1 + SWIR2)', {
        'SWIR1': img.select('S1'),
        'SWIR2': img.select('S2')
    }).rename('NBR2')

    BAI = img.expression('1 / ((0.1 - RED)**2 + (0.06 - NIR)**2)', {
        'RED': img.select('R'),
        'NIR': img.select('NIR')
    }).rename('BAI')

    MIRBI = img.expression('10 * SWIR2 - 9.8 * SWIR1 + 2', {
        'SWIR2': img.select('S2'),
        'SWIR1': img.select('S1')
    }).rename('MIRBI')

    return img.addBands([NDVI, EVI, BSI, IBI, SSI, RVI, DVI, GCVI, NBR, NBR2, BAI, MIRBI])


# 计算只有 Sentinel-2 可以计算的指数
def sentinel_indices(img):
    NBR_PLUS = img.expression('(SWIR2 - REDEDGE4 - GREEN - BLUE) / (SWIR2 + REDEDGE4 + GREEN + BLUE)', {
        'SWIR2': img.select('S2'),
        'REDEDGE4': img.select('RE4'),
        'GREEN': img.select('G'),
        'BLUE': img.select('B')
    }).rename('NBR+')

    BAIS2 = img.expression('(1 - sqrt((RE2 * RE3 * RE4) / R)) * ((S2 - RE4) / sqrt(S2 + RE4) + 1)', {
        'RE2': img.select('RE1'),
        'RE3': img.select('RE2'),
        'RE4': img.select('RE4'),
        'R': img.select('R'),
        'S2': img.select('S2')
    }).rename('BAIS2')

    REP = img.expression('705 + 35 * ((((NIRn1 + RED) / 2) - RE1) / (RE2 - RE1))', {
        'NIRn1': img.select('RE3'),
        'RE2': img.select('RE2'),
        'RE1': img.select('RE1'),
        'RED': img.select('R')
    }).rename('REP')

    return img.addBands([NBR_PLUS, BAIS2, REP])


# 计算 Landsat-8 可以计算的指数
def landsat_indices(img):
    REP = img.expression('705 + 35 * ((((NIRn1 + RED) / 2) - RE1) / (RE2 - RE1))', {
        'NIRn1': img.select('RE3'),
        'RE2': img.select('RE2'),
        'RE1': img.select('RE1'),
        'RED': img.select('R')
    }).rename('REP')

    return img.addBands([REP])


def calFVC(BestVI, region, scale):
    # 计算最小值和最大值
    num = BestVI.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=region,
        scale=scale,
        maxPixels=1e13
    )
    min_value = ee.Number(num.get("NDVI_p5"))
    max_value = ee.Number(num.get("NDVI_p95"))

    # 分位数和组合
    greater_part = BestVI.gt(max_value)
    less_part = BestVI.lt(min_value)
    middle_part = ee.Image(1).subtract(greater_part).subtract(less_part)

    # 计算 FVC
    tempf1 = BestVI.subtract(min_value).divide(max_value.subtract(min_value))

    FVC = ee.Image(1).multiply(greater_part) \
        .add(ee.Image(0).multiply(less_part)) \
        .add(tempf1.multiply(middle_part))

    return FVC.rename('FVC')


def load_Cols(target_time, coord):
    # 使用 ee.Date.parse 创建 ee.Date 对象，火前90天，火后60天无云合成
    middle_date = ee.Date.parse('YYYY/MM/dd', target_time)
    preDate = middle_date.advance(-60, 'day')
    postDate = middle_date.advance(45, 'day')
    # 创建点
    point = ee.Geometry.Point([coord['longitude'], coord['latitude']])
    roi = point.buffer(params['buffer']).bounds()
    # 调用loaddata函数，生成火灾前后的所有影像
    preFireS2, preFireS1, preFireL8 = loaddata(preDate, middle_date, roi)
    postFireS2, postFireS1, postFireL8 = loaddata(middle_date, postDate, roi)

    return preFireS2, preFireS1, preFireL8, postFireS2, postFireS1, postFireL8


# 添加id作为文件名称
@retry(tries=10, delay=1, backoff=2)
def getResult(point, image, name, id):
    # 检查图像是否已经导出
    if name not in exported_images:
        # point = ee.Geometry.Point(point['coordinates'])
        region = point.buffer(params['buffer']).bounds()
        if params['format'] in ['png', 'jpg']:
            url = image.getThumbURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )
        else:
            url = image.getDownloadURL(
                {
                    'region': region,
                    'dimensions': params['dimensions'],
                    'format': params['format'],
                }
            )
        if params['format'] == "GEO_TIFF":
            ext = 'tif'
        else:
            ext = params['format']
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()
        # 根据 name 的开头设置输出目录
        if name.startswith('pre'):
            out_dir = os.path.abspath(params['Pre'])
        elif name.startswith('post'):
            out_dir = os.path.abspath(params['Post'])
        else:
            out_dir = os.path.abspath(params['Pre'])  # 默认使用 'Pre' 目录
        # basename = str(index).zfill(len(str(params['count'])))
        # 原本pdate为params['prefix']
        filename = f"{out_dir}/{name}_{id}.{ext}"
        print(filename)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("Done: ", filename)
        # 将已导出图像的名称添加到列表中
        exported_images.append(name)
        return filename
    else:
        print('Image already exported:', name)
        return False


@retry(tries=10, delay=1, backoff=2)
def process_coord(coord):
    # inputdate = datetime.strptime(coord['time'], '%Y-%m-%d').strftime('%Y/%m/%d')
    inputdate = datetime.strptime(coord['time'], "%Y/%m/%d")

    # 查找最接近的两个时间点
    preFireS2, preFireS1, preFireL8, postFireS2, postFireS1, postFireL8 = load_Cols(coord['time'], coord)
    date_string = coord['time']
    # 将输入日期字符串解析为 datetime 对象
    input_date = datetime.strptime(date_string, "%Y/%m/%d")

    # 格式化为所需的字符串
    output_date_string = input_date.strftime("%Y%m%d")
    # 将点的时间日期添加为文件名称
    original_image_name = f"{coord['id']}_{output_date_string}"

    # 创建点
    point = ee.Geometry.Point([coord['longitude'], coord['latitude']])

    # 将点的时间日期添加为文件名称
    original_image_name = f"{'preFireS2'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, preFireS2, original_image_name, coord['id'])
    original_image_name = f"{'preFireS1'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, preFireS1, original_image_name, coord['id'])
    original_image_name = f"{'preFireL8'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, preFireL8, original_image_name, coord['id'])
    original_image_name = f"{'postFireS2'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, postFireS2, original_image_name, coord['id'])
    original_image_name = f"{'postFireS1'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, postFireS1, original_image_name, coord['id'])
    original_image_name = f"{'postFireL8'}_{output_date_string}"
    # 导出图像为 GeoTIFF，同时检查是否已经导出,传入id，作为文件名称命名
    success = getResult(point, postFireL8, original_image_name, coord['id'])

    # 将执行结果添加到列表中
    results_list.append([coord['longitude'], coord['latitude'], coord['time'],
                         'Success' if success else 'Failed', original_image_name])

if __name__ == "__main__":
# 提取经度、纬度和时间
    for index, row in df.iterrows():
        lon = row['Lon']
        lat = row['Lat']
        time = row['Date']
        id = row['Filename']

        # 检查是否已经处理过相同的日期和点位，判断时仅判断点位与日期。
        combo = f'{lon}_{lat}_{time}'
        if combo not in processed_combos:
            # 将经纬度和时间信息添加到列表中
            coordinates_list.append({
                'longitude': lon,
                'latitude': lat,
                'time': time,
                'id': id
            })

        # 将组合添加到已处理的列表中
        processed_combos.append(combo)
#     print(coordinates_list[0])
    tablesize = len(coordinates_list)
    print(tablesize)
    logging.basicConfig()
    # 使用 multiprocessing.Pool 创建进程池
    with tqdm(total=tablesize, desc='Download Files') as pbar:
        with Pool(params['processes']) as pool:
            for _ in pool.imap_unordered(process_coord, coordinates_list):
                pbar.update()
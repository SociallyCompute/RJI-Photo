# AVA_IMAGE_PATH = '/storage/hpc/group/augurlabs/images/'
AVA_IMAGE_PATH = '/media/matt/New Volume/ava/ava-compressed/images/'
MISSOURIAN_IMAGE_PATH = '/storage/hpc/group/augurlabs/2017/Fall/Dump/Cherryhomes, Ellie/20171025_Rockclimbing_ec'
# AVA_QUALITY_LABELS_FILE = '/storage/hpc/group/augurlabs/ava/AVA.txt'
AVA_QUALITY_LABELS_FILE = '/media/matt/New Volume/ava/AVA.txt'
# AVA_CONTENT_LABELS_FILE = '/storage/hpc/group/augurlabs/ava/tags.txt'
AVA_CONTENT_LABELS_FILE = '/media/matt/New Volume/ava/tags.txt'
# MODEL_STORAGE_PATH = '/storage/hpc/group/augurlabs/models/'
MODEL_STORAGE_PATH = '/College/research/RJI-Photo/models/'
DB_STR = 'postgresql://{}:{}@{}:{}/{}'.format(
        'rji', 'donuts', 'nekocase.augurlabs.io', '5433', 'rji'
    )

# Convert Missourian XMP Meta labels to standardized labels
XMP_AVA_MAP = {
    1: 10,
    2: 9,
    3: 7,
    4: 6,
    5: 5,
    6: 4,
    7: 2,
    8: 1
}
import os

import wget

def download_url_list(url_list, out):
    if not os.path.exists(out):
        os.makedirs(out)

    for u in url_list:
        print("\nDownloading from %s..." % u)
        wget.download(u, out)

if __name__ == '__main__':
    FSD_OUT_DIR = './FSDKaggle2018'

    FSD_URLS = [
        'https://zenodo.org/record/2552860/files/' + _
        for _ in ['FSDKaggle2018.audio_test.zip', 'FSDKaggle2018.audio_train.zip']
    ]

    TAU_OUT_DIR = './TAU-acoustic-sounds'

    TAU_DEV_URLS = [
        'https://zenodo.org/record/2589280/files/'
        'TAU-urban-acoustic-scenes-2019-development.audio.%d.zip' % i
        for i in range(1, 22)
    ]

    TAU_EVAL_URLS = [
        'https://zenodo.org/record/3063822/files/'
        'TAU-urban-acoustic-scenes-2019-evaluation.audio.%d.zip' % i
        for i in range(1, 12)
    ]

    download_url_list(FSD_URLS, FSD_OUT_DIR)
    download_url_list(TAU_DEV_URLS, TAU_OUT_DIR)
    download_url_list(TAU_EVAL_URLS, TAU_OUT_DIR)

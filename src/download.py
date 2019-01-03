import os
from google_drive_downloader import GoogleDriveDownloader as gdd

if __name__ == '__main__':

    # download all dataset
    print('downloading dataset ...')
    gdd.download_file_from_google_drive(file_id='10hxz4kxf9cnmoDC_hwBTLEVoguKptKWj',
                                            dest_path='./data/data.zip',
                                            unzip=True)
    # delete zip file
    os.remove('./data/data.zip')  
    print('Done.')
    
    # download models
    print('downloading models ...')
    gdd.download_file_from_google_drive(file_id='1JcdbExDVRYZPx4nzFTOkV2xzL1zSXZJJ',
                                            dest_path='./model/models.zip',
                                            unzip=True)
    # delete zip file
    os.remove('./model/models.zip')
    print('Done.')
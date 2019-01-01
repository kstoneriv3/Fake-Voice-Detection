import os
import argparse
from google_drive_downloader import GoogleDriveDownloader as gdd

if __name__ == '__main__':

    # download all dataset
    gdd.download_file_from_google_drive(file_id='1TqCBDRD6LFqgASS-kB8HOvCQKxkO4IBl',
                                            dest_path='./data/data.zip',
                                            unzip=True)
    # delete zip file
    os.remove('./data/data.zip')    
    
    # download models
    #gdd.download_file_from_google_drive(file_id='1ld84zHpPh4_kmKb27sQmV_RquxLSzANf',
    #                                        dest_path='./models.zip',
    #                                        unzip=True)
    # delete zip file
    #os.remove('./models.zip')
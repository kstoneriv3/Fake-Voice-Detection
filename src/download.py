import os
from google_drive_downloader import GoogleDriveDownloader as gdd

if __name__ == '__main__':

    # try to set the current working directory root of the project
    if os.getcwd().split('/')[-1]=='src' and os.getcwd().split('/')[-2]=='Fake-Voice-Detection':
        os.chdir('../')
        raise Warning('the working directory was changed to the root of this project.')

    if not os.getcwd().split('/')[-1]=='Fake-Voice-Detection':
        raise Exception('run the script at the root of this project.')

    # download all dataset
    gdd.download_file_from_google_drive(file_id='1ld84zHpPh4_kmKb27sQmV_RquxLSzANf',
                                            dest_path='./data.zip',
                                            unzip=True)
    
    # delete zip file
    os.remove('./data.zip')
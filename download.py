from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1ld84zHpPh4_kmKb27sQmV_RquxLSzANf',
                                    dest_path='./data/obama.zip',
                                    unzip=True)
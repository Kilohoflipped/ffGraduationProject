from app.libs.laz_process.dsm_generator import DSM

if __name__ == '__main__':
    dsm = DSM(laz_files_dir='./test/', step_size=0.09)
    # dsm.fill_no_data()
    dsm.save_dsm_image(file_path='test.tif')

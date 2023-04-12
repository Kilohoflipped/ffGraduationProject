from app.libs.laz_process.dsm_generator import DSM

if __name__ == '__main__':
    dsm = DSM(laz_files_path='E:/Kilo/Codes/Python/ffGraduationProject/app/data/lazs/points1.laz', step_size=0.1)
    dsm.save_dsm_image(file_path='test.tif')
    dsm.fill_no_data()
    dsm.save_dsm_image(file_path='testFilled.tif')

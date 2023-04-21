from app.libs.laz_process.dsm_generator import DSM
from app.libs.laz_process.CannyDetection import CannyDetection
from app.libs.laz_process.HoughTransform import HoughTransform

if __name__ == '__main__':
    # dsm = DSM(laz_files_path='E:/Kilo/Codes/Python/ffGraduationProject/app/data/lazs/points1.laz', step_size=0.1)
    # dsm.save_dsm_image(file_path='test.tif')
    # print("dsm generated")
    # dsm.fill_no_data()
    # print("missing filled")
    # dsm.save_dsm_image(file_path='testFilled.tif')
    # CannyDetection(picPath='E:/Kilo/Codes/Python/ffGraduationProject/app/testFilled.tif')
    # print("cannyed")
    # HoughTransform(picPath='E:/Kilo/Codes/Python/ffGraduationProject/app/testCannyed.tif')
    HoughTransform('无标题.tif')
    print("hougned")
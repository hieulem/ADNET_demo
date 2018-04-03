from shutil import copyfile
import argparse
import os.path
from m_util import *
opt = argparse.ArgumentParser().parse_args()
opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p500'
opt.train_fold = opt.im_fold+'_train/'
opt.test_fold = opt.im_fold+'_test/'
sdmkdir(opt.test_fold+'A')
sdmkdir(opt.test_fold+'B')
sdmkdir(opt.test_fold+'tif')
sdmkdir(opt.train_fold+'A')
sdmkdir(opt.train_fold+'tif')

imlist = []
for root,_,fnames in sorted(os.walk(opt.im_fold+'/A')):
    for fname in fnames:
        if fname.endswith('.png') and 'M1BS' in fname:
            imlist.append(fname)

print imlist
test_list = ["WV02_20160218213346_1030010053BC0300_16FEB18213346-M1BS-500638407020_01_P002_u08rf3031.png",
        "WV03_20160121015319_104001001762AC00_16JAN21015319-M1BS-500638671020_01_P001_u08rf3031.png",
        "WV03_20160202130111_1040010017D54900_16FEB02130111-M1BS-500638661010_01_P001_u08rf3031.png",
        "orthoWV02_11FEB111306175-M1BS-10300100080D2800_u08rfAEAC.png"
        ]

for im in imlist:
    if im not in test_list:
        copyfile(opt.im_fold+'/A/'+im,opt.train_fold+'A/'+im)
        copyfile(opt.im_fold+'/B/'+im,opt.train_fold+'B/'+im)
        copyfile(opt.im_fold+'/tif/'+im.replace('.png','.tif'),opt.train_fold+'tif/'+im.replace('.png','.tif')
    if im in test_list:
        copyfile(opt.im_fold+'/A/'+im,opt.test_fold+'A/'+im)
        copyfile(opt.im_fold+'/B/'+im,opt.test_fold+'B/'+im)
        copyfile(opt.im_fold+'/tif/'+im.replace('.png','.tif'),opt.test_fold+'tif/'+im.replace('.png','.tif')

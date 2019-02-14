rm -r train2014
rm -r val2014
mkdir train2014
mkdir val2014
python json2coco_crop_flip_region_classify1.py

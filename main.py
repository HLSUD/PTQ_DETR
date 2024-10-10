
# create quantized DETR based models

# load detection datasets
d_loader = COCOLoaderGenerator()
g=datasets.ViTImageNetLoaderGenerator(,'imagenet',32,32,16, kwargs={"model":net})
test_loader=g.test_loader()
calib_loader=g.calib_loader(num=calib_size)
# calibration

# evaluation
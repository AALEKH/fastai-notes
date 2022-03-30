```
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
* Above mentioned code is going to download 10,000+ pictures of dogs and cats, it uses simple rule to differentiate dogs and cats using their filename.
* It is then going to download pre-trained model which already knows how to recognize various images
* Then train the model to recognize dogs and cats
* Then its going to valdiate how good it is at recognize cats and dogs on the images it has not seen before
* Go through all the training picture once, this can also be said as 1 epoch to learn how to recognize dogs and cats.

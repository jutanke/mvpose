import numpy as np
import urllib.request
import shutil
from os.path import join, isfile, isdir
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image


class BoundBoxExtractor:

    def __init__(self, tmp='/tmp'):
        assert isdir(tmp)
        modelf = join(tmp, 'resnet50_coco_best_v2.1.0.h5')
        if not isfile(modelf):
            print("\t[bbextract] cannot find weights... downloading them...")
            url = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.0/resnet50_coco_best_v2.1.0.h5'
            with urllib.request.urlopen(url) as res, open(modelf, 'wb') as f:
                shutil.copyfileobj(res, f)
            print("\t[bbextract] store weights:", modelf)

        # load retinanet model
        self.model = models.load_model(modelf, backbone_name='resnet50')

    def extract_bbs(self, images_):
        """ extract all
        :param images_: [n x h x w x 3]
        :return:
        """
        n = len(images_)
        images = None
        scale = 0
        for i, image in enumerate(images_):
            img = preprocess_image(image)
            img, scale = resize_image(img)
            if images is None:
                h, w, c = img.shape
                assert c == 3
                images = np.empty((n, h, w, 3), np.float32)

            images[i] = img

        boxes, scores, labels = self.model.predict_on_batch(images)

        # correct for image scale
        boxes /= scale

        results = []
        for i in range(n):
            result_on_image = []
            results.append(result_on_image)
            for box, score, label in zip(boxes[i], scores[i], labels[i]):
                # scores are sorted so we can break
                if score < 0.5:
                    break

                if label == 0:
                    result_on_image.append(np.append(box, score))

        return results

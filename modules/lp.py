"""rame Number=2520 Number of Objects=2 Vehicle_count=2 Person_count=0
sgie_class >>> 2
sgie_class >>> 3
Frame Number=2521 Number of Objects=2 Vehicle_count=2 Person_count=0
sgie_class >>> 2
sgie_class >>> 3
Frame Number=2522 Number of Objects=3 Vehicle_count=2 Person_count=1
sgie_class >>> 2
sgie_class >>> 3
"""

from openalpr import Alpr


class LicencePlateDetector:
    def __init__(self, lang="eu", conf="/etc/openalpr/openalpr.conf", runtime_data="/usr/share/openalpr/runtime_data"):
        self.alpr = Alpr(lang, conf, runtime_data)
        if not self.alpr.is_loaded():
            print("Error loading OpenALPR")
            sys.exit(1)
        self.alpr.set_top_n(10)
        self.alpr.set_default_region("fi")

    def recognize_license_plate(image, obj_meta, confidence, p_frame_n):
        #frame_number = random.randint(0, 100)
        rect_params = obj_meta.rect_params
        top = int(rect_params.top)
        left = int(rect_params.left)
        width = int(rect_params.width)
        height = int(rect_params.height)
        car_cutout = image[top:top+height, left:left+width, :]

        # if (SAVE_IMAGES):
        #     cv2.imwrite("{0}/{1}_{2}_image.jpg".format(APPLICATION_PATH,
        #                                                p_frame_n, obj_meta.object_id), car_cutout)

        lp_top = -1
        lp_left = -1
        lp_bottom = -1
        lp_right = -1

        alrp_output = self.alpr.recognize_ndarray(car_cutout)
        lp_detection = None

        if ('results' in alrp_output and len(alrp_output['results']) > 0):
            out_LP_array = []
            lp_detection = alrp_output['results'][0]
        return alrp_output, lp_detection

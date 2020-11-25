import sys
from openalpr.src.bindings.python.openalpr import Alpr


class LpDetector:
    def __init__(self, countery="eu", region="fi", config="/etc/openalpr/openalpr.conf", runtime_data="/usr/share/openalpr/runtime_data"):
        self.alpr = Alpr(countery, config, runtime_data)
        if not self.alpr.is_loaded():
            print("Error loading OpenALPR")
            sys.exit()
        else:
            print("Using OpenALPR " + self.alpr.get_version())
            self.alpr.set_top_n(7)
            self.alpr.set_default_region(region)
            self.alpr.set_detect_region(False)

    def alpr_frame(self, image, obj_meta, confidence, p_frame_n):
        rect_params = obj_meta.rect_params
        top = int(rect_params.top)
        left = int(rect_params.left)
        width = int(rect_params.width)
        height = int(rect_params.height)
        car_cutout = image[top:top+height, left:left+width, :]

        # if (SAVE_IMAGES):
        #     cv2.imwrite("{0}/{1}_{2}_image.jpg".format(APPLICATION_PATH,p_frame_n,obj_meta.object_id),car_cutout)

        lp_top = -1
        lp_left = -1
        lp_bottom = -1
        lp_right = -1
        alrp_output = self.alpr.recognize_ndarray(car_cutout)
        return alrp_output



from py_utils.Utils_Base import *
import base64

from snakefoot import Server, gen_uuid

import lssc_ml


Dir(path="../dir_user",init=True)
USER_DICT = {}



def gen_image_name(image_label, image_id):
    return ".".join([str(image_label),image_id,"jpeg"])

def write_image_to_dir(data, dd):
    image_base64 = data["image_base64"]
    _, image_base64 = image_base64.split(",")
    image_data = base64.b64decode(image_base64)
    
    if "image_label" in data.keys():
        image_label = data["image_label"]
    else:
        image_label = "label"

    image_id = gen_uuid()
    image_name = gen_image_name(image_label, image_id)
    dd.set(image_name, image_data)

    return image_id

def delet_image_from_dir(dd):
    dir_sample_shape.delete( gen_image_name(data["image_label"], data["image_id"]) )

class LSSC_Server():

    def __init__(self):
        self.index_page = "index.html"

    def hello(self, params):
        print params
        return {"fuck":"you"}

    def get_user_and_data(self, params):
        return USER_DICT[params["user_id"]], params["data"]

    def create_user(self, params):
        user_id = gen_uuid()

        dir_user = Dir(path="../dir_user/"+user_id)
        dir_sample_shape = Dir(path=dir_user.path_join("sample_shape"), mode="binary")
        dir_std_shape = Dir(path=dir_user.path_join("std_shape"), mode="binary")
        dir_predict_shape = Dir(path=dir_user.path_join("predict_shape"), mode="binary")

        USER_DICT[user_id] = {
            "dir_user": dir_user,
            "dir_sample_shape": dir_sample_shape,
            "dir_std_shape": dir_std_shape,
            "dir_predict_shape": dir_predict_shape,
        }

        return dict(user_id=user_id)

    def user_upload(self, params):
        user, data = self.get_user_and_data(params)
        
        dir_sample_shape = user["dir_sample_shape"]
        image_id = write_image_to_dir(data, dir_sample_shape)

        return {"image_id": image_id}

    def user_delete(self, params):
        user, data = self.get_user_and_data(params)

        dir_sample_shape = user["dir_sample_shape"]
        dir_sample_shape.delete( gen_image_name(data["image_label"], data["image_id"]) )

    def user_train(self, params):
        user, data = self.get_user_and_data(params)

        dir_sample_shape = user["dir_sample_shape"]
        
        if lssc_ml.check_dir_fit(dir_sample_shape):
            model = lssc_ml.fit_model_by_dir(dir_sample_shape)
            user["model"] = model
            return {"train_ok":True }
        else:
            return {"train_ok":False , "warning":"Training Error! Please ensure the sample correct"}

    def user_predict(self, params):
        user, data = self.get_user_and_data(params)

        dir_predict_shape = user["dir_predict_shape"]
        image_id = write_image_to_dir(data, dir_predict_shape)

        if "model" in user.keys():
            model = user["model"]
        else:
            return {"predict_ok": False}

        predict_label = lssc_ml.predict_by_model(model, dir_predict_shape)

        dir_predict_shape.init_dir()

        return {"predict_label": predict_label, "predict_ok": True}


if __name__ == '__main__':
    s = Server(LSSC_Server())
    s.bind(host="0.0.0.0", port=10088, debug=False)
    s.run()



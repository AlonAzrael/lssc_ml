

function LSSC_Client_C () {

    var client = snakefoot.Client()
    // client.connect("http://") 

    function wrap_data (data) {
        return {user_id: USER_ID, data: data}
    }

    var lssc_client = {
        create_user: function () {
            client.invoke("create_user", {}, function (result) {
                USER_ID = result.user_id
            })
        },

        user_upload: function (dataURL, label, callback) {
            var data = {image_base64:dataURL, image_label:label}
            client.invoke("user_upload", wrap_data(data), callback)
        },

        user_delete: function (label, image_id, callback) {
            var data = {image_label:label, image_id:image_id}
            client.invoke("user_delete", wrap_data(data), callback)
        },

        user_train: function () {
            client.invoke("user_train", wrap_data({}), function (result) {
                if (!result.train_ok) {
                    alert(result.warning)
                }
                else {
                    print(result)
                }
            })
        },

        user_predict: function (dataURL, callback) {
            var data = {image_base64:dataURL}
            client.invoke("user_predict", wrap_data(data), function (result) {
                if (result.predict_ok) {
                    callback(result)
                }
                else {
                    alert("Nothing to predict yet!")
                }
            })
        },

    }

    return lssc_client
}

LSSC_CLIENT = LSSC_Client_C()



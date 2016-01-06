
function print (argument) {
    console.log.apply(console, arguments)
}

function send_to_server (url, req_obj, callback) {
    var req_data = JSON.stringify(req_obj)
    $.ajax({
        method:"POST",
        url: url,
        contentType: "application/json; charset=UTF-8",
        data: req_data,
        dataType: "json",
    })
    .done(function(result) {
        if (result["status"]) {
            if (callback!=undefined) {
                callback(result["result"])
            }
        } 
        else {
            print(result["msg"])
        }
    })
    .fail(function () {
        print("send failed")
    })
}

function snakefoot_Client_C () {
    
    var snakefoot_Client = {
        invoke: function (callback_name, callback_params, result_callback) {
            if (callback_params==undefined) {
                callback_params = {}
            }
            send_to_server("/__snakefoot_rpc_invoke", {callback_name:callback_name, callback_params:callback_params,
            }, result_callback)
        },
    }

    return snakefoot_Client
}

var snakefoot = {
    Client: function () {
        return snakefoot_Client_C()
    },
}






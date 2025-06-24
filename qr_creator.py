import qrcode
import json

data = {
    "orderId": "16",
    "modelVersionId": "9"
}

json_data = json.dumps(data)

qr = qrcode.make(json_data)
qr.save("qr_order.png")

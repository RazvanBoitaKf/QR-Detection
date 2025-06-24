import cv2
from qreader import QReader
import json

qreader = QReader()

img = cv2.imread('qr_order.png')

bboxes = qreader.detect(image=img)

for bbox in bboxes:
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox['bbox_xyxy'])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_qr = img[y1:y2, x1:x2]
        decoded = qreader.detect_and_decode(cropped_qr)

        for data in decoded:
            if data:
                print("Raw QR data:", data)
                try:
                    parsed = json.loads(data)
                    print("Order ID:", parsed.get("orderId"))
                    print("Model Version ID:", parsed.get("modelVersionId"))
                except json.JSONDecodeError:
                    print("Not valid JSON")

cv2.imshow("Detected QR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
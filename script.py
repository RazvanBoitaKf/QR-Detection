import cv2

# 0 is usually the default webcam; change to 1,2,... if needed
CAM_INDEX = 0

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame (stream end?). Exiting...")
        break

    cv2.imshow("Raw Camera Feed", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

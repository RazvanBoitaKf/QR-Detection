import cv2
import time
import os
import csv
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient
from qreader import QReader
import json

IMAGE_PATH = "dataset/captured_image.jpg"
LOG_PATH = "log.csv"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3N9lBuccbpGSuITSmuP8"
)

def run_inference(path):
    result = client.run_workflow(
        workspace_name="my-workspace-2eheg",
        workflow_id="custom-workflow-4",
        images={"image": path},
        use_cache=True
    )
    return result

def interpret_result(result):
    if not result or not result[0]['predictions']['predictions']:
        return None, 0.0
    prediction = result[0]['predictions']['predictions'][0]
    class_name = prediction.get('class')
    confidence = prediction.get('confidence', 0.0)

    if class_name == "4":
        class_name = "wrinkles"

    return class_name, confidence

def save_log(class_name, confidence):
    with open(LOG_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), class_name, f"{confidence:.4f}"])

class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Image Classifier")
        self.root.geometry("800x800")
        self.root.resizable(False, False)

        self.label = tk.Label(root, text="Live Camera Feed", font=("Arial", 22))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.status_label = tk.Label(root, text="", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=5)

        self.capture_btn = ttk.Button(root, text="Capture & Predict", command=self.capture_and_predict)
        self.capture_btn.pack(pady=10)

        self.upload_btn = ttk.Button(root, text="Upload Image & Predict", command=self.upload_and_predict)
        self.upload_btn.pack(pady=5)

        self.video_capture = cv2.VideoCapture(0)
        self.feed_running = True
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            root.destroy()
        else:
            self.update_frame()

    def update_frame(self):
        if self.feed_running:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_and_predict(self):
        self.capture_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...", fg="blue")
        self.result_label.config(text="")
        self.feed_running = False
        threading.Thread(target=self.process_prediction).start()

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.capture_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing uploaded image...", fg="blue")
        self.result_label.config(text="")
        self.feed_running = False

        threading.Thread(target=self.process_uploaded_image, args=(file_path,)).start()

    def process_prediction(self):
        try:
            os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
            cv2.imwrite(IMAGE_PATH, self.current_frame)

            result = run_inference(IMAGE_PATH)
            class_name, confidence = interpret_result(result)
            annotated_frame = self.current_frame.copy()

            try:
                predictions = result[0]['predictions']['predictions']
                for pred in predictions:
                    x_center, y_center = pred['x'], pred['y']
                    width, height = pred['width'], pred['height']
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{pred['class']} ({pred['confidence']:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except Exception as e:
                print("Bounding box error:", e)

            try:
                qreader = QReader()
                bboxes = qreader.detect(image=self.current_frame)

                for bbox in bboxes:
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox['bbox_xyxy'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cropped_qr = self.current_frame[y1:y2, x1:x2]
                        decoded = qreader.detect_and_decode(cropped_qr)
                        for data in decoded:
                            if data:
                                print("QR:", data)
                                try:
                                    parsed = json.loads(data)
                                    qr_text = f"QR: Order ID {parsed.get('orderId')}, Version {parsed.get('modelVersionId')}"
                                    cv2.putText(annotated_frame, qr_text, (x1, y2 + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                except json.JSONDecodeError:
                                    print("Invalid JSON in QR.")
            except Exception as e:
                print("QR detection error:", e)

            cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((800, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

            if class_name:
                result_text = f"Prediction: {class_name} ({confidence:.2%})"
                self.result_label.config(text=result_text, fg="green")
                save_log(class_name, confidence)
            else:
                self.result_label.config(text="No prediction made.", fg="orange")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_label.config(text="Error occurred.", fg="red")
        finally:
            self.status_label.config(text="")
            time.sleep(2.5)
            self.feed_running = True
            self.capture_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)

    def process_uploaded_image(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError("Could not load image.")

            self.current_frame = frame
            result = run_inference(image_path)
            class_name, confidence = interpret_result(result)
            annotated_frame = frame.copy()

            try:
                predictions = result[0]['predictions']['predictions']
                for pred in predictions:
                    x_center, y_center = pred['x'], pred['y']
                    width, height = pred['width'], pred['height']
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{pred['class']} ({pred['confidence']:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except Exception as e:
                print("Bounding box error (uploaded):", e)

            try:
                qreader = QReader()
                bboxes = qreader.detect(image=frame)

                for bbox in bboxes:
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox['bbox_xyxy'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cropped_qr = frame[y1:y2, x1:x2]
                        decoded = qreader.detect_and_decode(cropped_qr)
                        for data in decoded:
                            if data:
                                print("QR from uploaded image:", data)
                                try:
                                    parsed = json.loads(data)
                                    qr_text = f"QR: Order ID {parsed.get('orderId')}, Version {parsed.get('modelVersionId')}"
                                    cv2.putText(annotated_frame, qr_text, (x1, y2 + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                except json.JSONDecodeError:
                                    print("Invalid JSON in QR from upload.")
            except Exception as e:
                print("QR detection error (uploaded):", e)

            cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((800, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

            if class_name:
                result_text = f"Prediction: {class_name} ({confidence:.2%})"
                self.result_label.config(text=result_text, fg="green")
                save_log(class_name, confidence)
            else:
                self.result_label.config(text="No prediction made.", fg="orange")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_label.config(text="Error occurred.", fg="red")
        finally:
            self.status_label.config(text="")
            time.sleep(2.5)
            self.feed_running = True
            self.capture_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)

    def on_close(self):
        self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)

    root = tk.Tk()
    app = InferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

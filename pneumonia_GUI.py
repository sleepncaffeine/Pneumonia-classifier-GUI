# import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from ttkthemes import ThemedTk
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk


def classify_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.resize(image, (200, 200))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        try:
            result_label.config(text="분석 중...(1/5)")
            root.update()
            pred1 = model1.predict(image)
            result_label.config(text="분석 중... (2/5)")
            root.update()
            pred2 = model2.predict(image)
            result_label.config(text="분석 중... (3/5)")
            root.update()
            pred3 = model3.predict(image)
            result_label.config(text="분석 중... (4/5)")
            root.update()
            pred4 = model4.predict(image)
            result_label.config(text="분석 중... (5/5)")
            root.update()
            pred5 = model5.predict(image)
            result_label.config(text="분석 완료")
            root.update()

            p_pred1 = pred1.T[0]
            p_pred2 = np.array([x[0] for x in pred2])
            p_pred3 = pred3.T[0]
            p_pred4 = np.array([x[0] for x in pred4])
            p_pred5 = np.array([x[0] for x in pred5])

            preds = [p_pred1, p_pred2, p_pred3, p_pred4, p_pred5]

            weights = [
                0.22307305766017074,
                0.28895263929699505,
                0.11785743965075406,
                0.37011686339208016,
                0.0,
            ]

            model_num = 5
            data_size = len(p_pred1)

            ensemble_pred = np.array(
                [
                    sum([weights[i] * preds[i][j] for i in range(model_num)])
                    for j in range(data_size)
                ]
            )

            is_pneumonia = ensemble_pred[0] < 0.5

            # confidence_normal = round(ensemble_pred[0] * 100, 2)
            # confidence_pneumonia = 100 - confidence_normal

            if is_pneumonia:
                result = "폐렴"
                # display_text = "{}% 확률로 {}입니다.".format(confidence_pneumonia, result)
                display_text = "{}입니다.".format(result)
            else:
                result = "정상"
                # display_text = "{}% 확률로 {}입니다. (폐렴일 확률: {}%)".format(confidence_normal, result, confidence_pneumonia)
                display_text = "{}입니다.".format(result)

            result_label.config(text=display_text)
            root.update()

        except Exception as e:
            print(e)
            result_label.config(text="오류가 발생했습니다.")
            root.update()


def remove_result():
    image_label.config(image="")
    result_label.config(text="")
    root.update()


# root = tk.Tk()
root = ThemedTk(theme="arc")
root.title("폐렴 진단")
root.geometry("540x450")
icon = ImageTk.PhotoImage(file="icon/free-icon-infected-lungs-2853828.png")
root.iconphoto(False, icon)

loading_label = ttk.Label(
    root,
    text="로딩 중...",
    font=("맑은 고딕", 15),
)
loading_label.place(relx=0.5, rely=0.5, anchor="center")
root.update()

model1 = load_model("models/model1.h5")
model2 = load_model("models/model2.h5")
model3 = load_model("models/model3.h5")
model4 = load_model("models/model4.h5")
model5 = load_model("models/model5.h5")

loading_label.destroy()

description_label = ttk.Label(
    root,
    text="폐렴 진단 어플리케이션\n흉부 X-ray 이미지를 선택하면 폐렴인지 아닌지 진단해줍니다.",
    font=("맑은 고딕", 12),
)
description_label.pack(side="top")


put_image_button = ttk.Button(
    root,
    text="이미지 선택",
    command=classify_image,
    padding=(10, 5),
)
put_image_button.pack(pady=10)

image_label = ttk.Label(root)
image_label.pack()

result_label = ttk.Label(root, font=("맑은 고딕", 14))
result_label.pack(pady=10)

information_label = ttk.Label(
    root,
    text="주의: 이 어플리케이션은 특별한 의학적 조언을 대신하지 않습니다.",
    font=("맑은 고딕", 10),
)
information_label.pack(side="bottom")

remove_result_button = ttk.Button(
    root,
    text="결과 제거",
    command=remove_result,
    padding=(10, 5),
)
remove_result_button.pack(side="bottom", pady=10)

root.mainloop()

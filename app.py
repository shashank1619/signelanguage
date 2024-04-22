from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from tensorflow import keras
import operator
from string import ascii_uppercase


class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.json_file = open("model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model-bw.h5")

        self.json_file_dru = open("model-bw_dru.json", "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("model-bw_dru.h5")

        self.json_file_tkdi = open("model-bw_tkdi.json", "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("model-bw_tkdi.h5")

        self.json_file_smn = open("model-bw_smn.json", "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("model-bw_smn.h5")

        self.ct = {}
        self.ct["blank"] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("900x1100")

        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)
        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=460, y=95, width=310, height=310)

        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current SYmbol
        self.panel3.place(x=450, y=640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Character :", font=("Courier", 20, "bold"))

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=180, y=680)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=680)
        self.T2.config(text="Word :", font=("Courier", 20, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=220, y=720)
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=720)
        self.T3.config(text="Sentence :", font=("Courier", 20, "bold"))

        self.btcall = tk.Button(self.root, command=self.action_call, height=0, width=0)
        self.btcall.config(text="About", font=("Courier", 14))
        self.btcall.place(x=825, y=0)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 10, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            ret, res = cv2.threshold(
                th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 20))
            self.panel4.config(text=self.word, font=("Courier", 20))
            self.panel5.config(text=self.str, font=("Courier", 20))

        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
        result_digit = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}
        prediction["blank"] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        # LAYER 1
        prediction = sorted(
            prediction.items(), key=operator.itemgetter(1), reverse=True
        )
        self.current_symbol = prediction[0][0]
        # LAYER 2
        if (
            self.current_symbol == "D"
            or self.current_symbol == "R"
            or self.current_symbol == "U"
        ):
            prediction = {}
            prediction["D"] = result_dru[0][0]
            prediction["R"] = result_dru[0][1]
            prediction["U"] = result_dru[0][2]
            # prediction['1'] = result_digit[0][0]

            prediction = sorted(
                prediction.items(), key=operator.itemgetter(1), reverse=True
            )
            self.current_symbol = prediction[0][0]

        if (
            self.current_symbol == "D"
            or self.current_symbol == "I"
            or self.current_symbol == "K"
            or self.current_symbol == "T"
        ):
            prediction = {}
            prediction["D"] = result_tkdi[0][0]
            prediction["I"] = result_tkdi[0][1]
            prediction["K"] = result_tkdi[0][2]
            prediction["T"] = result_tkdi[0][3]
            prediction = sorted(
                prediction.items(), key=operator.itemgetter(1), reverse=True
            )
            self.current_symbol = prediction[0][0]
        if (
            self.current_symbol == "M"
            or self.current_symbol == "N"
            or self.current_symbol == "S"
        ):
            prediction1 = {}
            prediction1["M"] = result_smn[0][0]
            prediction1["N"] = result_smn[0][1]
            prediction1["S"] = result_smn[0][2]
            prediction1 = sorted(
                prediction1.items(), key=operator.itemgetter(1), reverse=True
            )
            if prediction1[0][0] == "S":
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]
        if self.current_symbol == "blank":
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if self.ct[self.current_symbol] > 40:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct["blank"] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct["blank"] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == "blank":
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self):

        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol("WM_DELETE_WINDOW", self.destructor1)
        self.root1.geometry("900x900")

        self.tx = tk.Label(self.root1)
        self.tx.place(x=330, y=20)
        self.tx.config(text="Efforts By", fg="red", font=("Courier", 30, "bold"))

        # self.photo1 = tk.PhotoImage(file='Pictures/Ankush.jpg')
        self.w1 = tk.Label(self.root1)
        self.w1.place(x=20, y=105)
        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x=50, y=250)
        self.tx6.config(text="Shashank kumar \n 2104186", font=("Courier", 15, "bold"))

        # self.photo2 = tk.PhotoImage(file='Pictures/Abhishek.png'),, image = self.photo2
        self.w2 = tk.Label(self.root1)
        self.w2.place(x=200, y=105)
        self.tx2 = tk.Label(self.root1)
        self.tx2.place(x=290, y=250)
        self.tx2.config(text="Rahul sharma \n 2104110", font=("Courier", 15, "bold"))

        # self.photo3 = tk.PhotoImage(file='Pictures/Rajat.png')
        self.w3 = tk.Label(self.root1)
        self.w3.place(x=380, y=105)
        self.tx3 = tk.Label(self.root1)
        self.tx3.place(x=490, y=250)
        self.tx3.config(text="Saneh Thakur \n 2104180", font=("Courier", 15, "bold"))

        self.tx7 = tk.Label(self.root1)
        self.tx7.place(x=170, y=360)
        self.tx7.config(
            text="Under the supervision of", fg="red", font=("Courier", 25, "bold")
        )

        # self.photo6 = tk.PhotoImage(file='Pictures/sir.png')
        self.w6 = tk.Label(self.root1)
        self.w6.place(x=350, y=420)
        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x=230, y=670)
        self.tx6.config(text="Prof. SUPREET KAUR", font=("Courier", 20, "bold"))


print("Starting Application...")
mainApp = Application()
mainApp.root.mainloop()

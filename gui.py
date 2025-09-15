import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import cv2
import keras
import numpy as np

from keras.models import load_model # type: ignore
model = keras.models.load_model('C:/Users/bayya/Desktop/dum/minor project/lung_cancer_detection_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classes=['lung_aca', 'lung_scc', 'lung_n']
IMG_SIZE = 256



top=tk.Tk()
top.geometry('800x600')
top.title('lung cancer detection')
top.configure(background='#CDCDCD')

label1=Label(top,background='#CDCDCD',font=('arial',15,'bold'),text='')
sign_image=Label(top,text="image")



# Making Predictions.....
def detect(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File {image_path} does not exist")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}")

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(classes[predicted_class])
    label1.configure(foreground='#011638',text=classes[predicted_class])





def show_detect_button(file_path):
    detect_button=Button(top,text='detect image',command=lambda: detect(file_path),padx=5,pady=5)
    detect_button.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
    detect_button.place(relx=0.70,rely=0.27)

def upload_image():
    try:
        file_path=filedialog.askopenfile()
        uploaded=Image.open(file_path.name)
        uploaded.thumbnail(((top.winfo_width()/2.1),(top.winfo_height()/2.1)))
        image=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=image)
        sign_image.image=image
        label1.configure(text='')
        show_detect_button(file_path.name)
    except:
        pass







upload=Button(top,text='upload image',command=upload_image,padx=2,pady=2)
upload.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
sign_image.pack(side='bottom',expand=True)
upload.pack(side='bottom')


label1.pack(side='bottom',pady=1,expand=True)
heading=Label(top,text="lung cancer detection",pady=5,font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

from tkinter import *
from PIL import Image, ImageTk
import x_train
from threading import Thread
from graph import *
from PIL import Image, ImageTk
def x_main():


    def train():
        num = scale.get()
        print(num)
        x_train.Train_x(num)
        # window.showinfo("训练完成")
        print(1)



    window = Tk()
    window.geometry('%dx%d' % (500, 300))
    window.title('y = x ^ 2 ')
    scale = Scale(window, from_=100, to=1000, orient=HORIZONTAL,
                  tickinterval=100, length=200, resolution=100)
    scale.pack()
    # window.showinfo(title='错误', message='账号或密码错误！')

    Button(window, text="训练", command=train).pack()
    Button(window, text='泛化曲线', command=show_generation_graph).pack()
    Button(window, text='回想曲线', command=show_recall_graph).pack()

    Button(window, text='训练曲线', command=show_train_graph).pack()


    window.mainloop()

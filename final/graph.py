from tkinter import *
from PIL import Image, ImageTk

def show_recall_graph():
    graph1 = Toplevel()
    graph1.title('回想曲线')
    img_open = Image.open('recall.jpg')
    imge_png = ImageTk.PhotoImage(img_open)
    label1 = Label(graph1, image=imge_png)
    label1.pack()
    graph1.mainloop()

def show_generation_graph():
    graph2 = Toplevel()
    graph2.title('泛化曲线')
    img_open = Image.open('generation.jpg')
    imge_png = ImageTk.PhotoImage(img_open)
    label1 = Label(graph2, image=imge_png)
    label1.pack()


    graph2.mainloop()

def show_train_graph():
    graph3 = Toplevel()
    graph3.title('训练曲线')
    img_open = Image.open('train.jpg')
    imge_png = ImageTk.PhotoImage(img_open)
    label1 = Label(graph3, image=imge_png)
    label1.pack()

    graph3.mainloop()


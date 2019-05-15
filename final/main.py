from tkinter import *
from func_sin import sin_main
from func_xx import x_main
from func_12 import x12_main

def main():
    window = Tk()
    window.title('BP神经网络')
    window.geometry('%dx%d' % (500, 300))  # 设置窗口大小

    # MainPage(window)
    Button(text='y  = sin(x)', command=sin_main).grid(row=1, column=4, stick=W, pady=10)

    Button(text='y = x^2', command=x_main).grid(row=2, column=4, stick=W, pady=10)
    Button(text='y = x1 + x2 ', command=x12_main).grid(row=3, column=4, stick=W, pady=10)
    Button(text='退出', command=window.quit).grid(row=4, column=4, pady=10, stick=W)

    window.mainloop()


main()

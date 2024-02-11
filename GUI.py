from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
import time
from PIL import ImageTk, Image

# window
window = Tk()
WINDOW_GEOMETRY = window.geometry('950x730')
window.resizable(0, 0)
window.title('Tumor Cancer Prediction')
window.configure(bg='dark turquoise')

# uploadfile
def open_file():
    file_path = askopenfile(mode='r', filetypes=[('Image Files', '*jpeg')])
    if file_path is not None:
        pass


def uploadFiles():
    pb1 = Progressbar(
        ws,
        orient=HORIZONTAL,
        length=300,
        mode='determinate'
    )
    pb1.grid(row=4, columnspan=3, pady=20)
    for i in range(5):
        ws.update_idletasks()
        pb1['value'] += 20
        time.sleep(1)
    pb1.destroy()
    Label(ws, text='File Uploaded Successfully!', foreground='green').grid(row=4, columnspan=3, pady=10)


adhar = Label(
    window,
    text='Upload File '
)
adhar.grid(row=0, column=0, padx=10)

adharbtn = Button(
    window,
    text='Browse',
    command=lambda: open_file()
)
adharbtn.grid(row=0, column=1)

window.mainloop()

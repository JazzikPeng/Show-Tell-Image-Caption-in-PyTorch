from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import time
import os
from attention_caption import get_caption
from our_caption import generate_caption, Vocabulary


class App:

    def __init__(self, root):
        root.title("Caption Generator")
        """
        Frame configurations:
            height, width: int (pixles)
            relief: ["FLAT", "RAISED", "SUNKEN", "SOLID", "RIDGE", "GROOVE "]
        layout configurations:
            pack:
                side: LEFT/RIGHT/TOP/BUTTOM
        """
        self.panedwindow = ttk.Panedwindow(root, orient=HORIZONTAL)
        self.panedwindow.pack(fill=BOTH, expand=True)

        self.frame_input = ttk.Frame(self.panedwindow, heigh=512, width=512,
                                     relief=FLAT)
        self.frame_input.grid(row=0, column=1, columnspan=4)

        self.load_model_button = ttk.Button(self.frame_input,
                                            command=self.load_model,
                                            text="Select Model")
        self.load_model_button.grid(row=1, column=0)

        self.load_image_buttion = ttk.Button(self.frame_input,
                                             command=self.load_image,
                                             text="Select Image")
        self.load_image_buttion.grid(row=1, column=1)
        self.generate_caption_button = ttk.Button(self.frame_input,
                                                  command=self.genearte_caption,
                                                  text="Generate Caption")
        self.generate_caption_button.grid(row=1, column=2)
        self.reset_button = ttk.Button(self.frame_input,
                                       command=self.reset,
                                       text="Reset")
        self.reset_button.grid(row=1, column=3)

        self.frame_output = ttk.Frame(self.panedwindow, heigh=512, width=512,
                                      relief=FLAT)
        self.frame_output.grid(row=1, column=1, columnspan=3)

        self.caption_container = ttk.Frame(self.panedwindow, heigh=8, width=512,
                                           relief=FLAT)
        self.caption_container.grid(row=2, column=0, columnspan=3)
        self.exec_container = ttk.Label(self.caption_container, text="Waiting for an image input!")
        self.exec_container.grid(row=2, column=0, columnspan=3)

        self.frame_status = ttk.Frame(self.panedwindow, heigh=8, width=512,
                                      relief=FLAT)
        self.frame_status.grid(row=3, column=0, columnspan=3)

        self.exec_status = ttk.Label(self.frame_status, text="READY TO RECIEVE COMMAND!")
        self.exec_status.grid(row=3, column=0, columnspan=3)

        self.logo = ttk.Label(self.frame_output)
        self.model_path_default = "./data/our_model.p"  # BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar
        self.model_status = "our"
        self.model_path = None
        self.img_path = None

    def load_model(self):
        self.model_path = filedialog.askopenfile()
        self.exec_status.config(text="Load the Model from: " +
                                os.path.basename(
                                    self.model_path.name))
        if self.model_path != "./data/our_model.p":
            self.model_status = "attention"

    def load_image(self):
        self.img_path = filedialog.askopenfile()
        self.exec_status.config(text="Load the Image from: " +
                                os.path.basename(
                                    self.img_path.name))
        self.logo.img = ImageTk.PhotoImage(Image.open(self.img_path.name))
        self.logo.config(image=self.logo.img)
        self.logo.grid(row=3, column=0)
        self.exec_container.config(text="You can generate caption now!")

    def genearte_caption(self):
        self.exec_status.config(text="BUSY!")
        self.exec_status.update()
        self.exec_container.config(text="Generating ... Please Wait!")
        self.exec_container.update()
        # time.sleep(5)
        if self.img_path is not None:
            if self.model_path is None:
                self.exec_container.config(text="Using model with {}... Please Wait!".format(self.model_status))
                self.exec_container.update()
                if self.model_status == "attention":
                    _, caption = get_caption(self.img_path.name, self.model_path_default)
                else:
                    caption = generate_caption(self.img_path.name, self.model_path_default)
            else:
                if self.model_status == "attention":
                    _, caption = get_caption(self.img_path.name, self.model_path.name)
                else:
                    caption = generate_caption(self.img_path.name, self.model_path.name)
            self.exec_container.config(text=caption)
            print(caption)
            self.exec_status.config(text="READY TO RECIEVE COMMAND!")
        else:
            self.exec_container.config(text="Please provide a image!")
            self.exec_status.config(text="READY TO RECIEVE COMMAND!")

    def reset(self):
        self.exec_container.config(text="Waiting for an image input!")
        self.logo.config(image="")
        if self.img_path is not None:
            self.img_path = None


def exec():
    root = Tk()
    style = ttk.Style()
    style.theme_use('aqua')  # ('aqua', 'clam', 'alt', 'default', 'classic')
    App(root)
    root.mainloop()


if __name__ == "__main__":
    exec()

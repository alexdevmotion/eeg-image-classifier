from Tkinter import *
from PIL import ImageTk, Image
import tkMessageBox
import sys
import os


def getNoImagesInDirectory(dir):
    return len(getImagesInDirectory(dir))


def getImagesInDirectory(dir):
    files = os.listdir(dir)
    images = []
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".gif")):
            images.append(file)
    return images


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class FullScreenWindow:
    def __init__(self, closingCallback):
        self.closingCallback = closingCallback

        self.tk = Tk()
        self.frame = Frame(self.tk)
        self.frame.pack()
        self.state = False
        self.tk.iconbitmap("misc/favicon.ico")
        self.tk.title("EEG Unified Logger a.k.a. The Experiment Machine")
        self.tk.minsize(width=600, height=400)

        self.tk.bind("<F11>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)

        self.tk.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"

    def on_closing(self):
        if tkMessageBox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.tk.destroy()
            if self.closingCallback:
                self.closingCallback()
            sys.exit(0)


class ImageWindow:
    def __init__(self, dir, images, imageInterval, threadedTasks, crop):
        self.dir = dir
        self.images = images
        self.imageInterval = imageInterval
        self.threadedTasks = threadedTasks
        self.crop = crop

        self.curImageIndex = 0

    def tkAndWindow(self, tk):
        self.tk = tk
        self.window = Toplevel(self.tk)
        self.window.attributes("-fullscreen", True)
        self.window.focus_force()
        self.window.bind("<Escape>", self.experimentStoppedByUser)
        self.windowDestroyed = False

        self.imagePanel = Label(self.window, image=None)
        self.imagePanel.pack(side="bottom", fill="both", expand="yes")

    def experimentStoppedByUser(self, event=None):
        self.window.destroy()
        self.windowDestroyed = True
        self.threadedTasks.stopLogging()

    def handleNextImage(self, keep_going=True):
        if not self.windowDestroyed:
            try:
                curImage = str(self.images[self.curImageIndex])
                self.threadedTasks.setCurrentFileName(curImage)
                self.displayImage(self.dir + "/" + curImage)
                self.curImageIndex += 1
                if keep_going:
                    self.window.after(self.imageInterval * 1000, self.handleNextImage)
                else:
                    self.window.after(self.imageInterval * 1000, self.experimentStoppedByUser)
            except IndexError:
                self.experimentStoppedByUser()
                return False
        return True

    def displayImage(self, path):
        img = Image.open(path)
        if self.crop:
            img = self.cropAndResize(img, self.window.winfo_screenwidth(), self.window.winfo_screenheight())

        photoimg = ImageTk.PhotoImage(img)

        self.imagePanel.configure(image=photoimg)
        self.imagePanel.image = photoimg

    def cropAndResize(self, image, ideal_width, ideal_height):
        width  = image.size[0]
        height = image.size[1]

        aspect = width / float(height)

        ideal_aspect = ideal_width / float(ideal_height)

        if aspect > ideal_aspect:
            # Then crop the left and right edges:
            new_width = int(ideal_aspect * height)
            offset = (width - new_width) / 2
            resize = (offset, 0, width - offset, height)
        else:
            # ... crop the top and bottom:
            new_height = int(width / ideal_aspect)
            offset = (height - new_height) / 2
            resize = (0, offset, width, height - offset)

        return image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)

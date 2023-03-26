import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

x1 = y1 = x2 = y2 = 0

def line_select_callback(eclick, erelease):
    global x1,y1,x2,y2
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)

def user_drawn_boundary(img):
    fig, current_ax = plt.subplots()
    img=img[:,:,::-1]
    plt.imshow(img)
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback, useblit=True, interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    return (x1,y1,x2,y2)

def get_user_boundary():
    return (x1,y1,x2,y2)


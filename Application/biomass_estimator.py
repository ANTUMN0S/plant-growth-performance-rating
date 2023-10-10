'''
This will be the main programm that will be run by the user. It will call upon all the functions necessary to generate a weight estimate form a basil picture:
1: Import the unedited picture
2: Segment the picture with the pretrained CNN (or SAM if we don't get it to work), then use the generated mask to continue working
3: From the Masks, get the pot diameter, capture angle and plant area
4: Using a pretrained model, apply linear regression to get weight estimate
5: Print weight (and Feedback, optionally)
'''
import tkinter as tk
from tkinter import filedialog
import tkinterDnD
from tkdnd import DND_FILES
from PIL import Image, ImageTk
import screeninfo as si
import threading
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# local imports
from util import weight_estimation, segmentation_lang, segmentation_by_coordinates, make_dataframe_from_array

# Global variables to store clicked coordinates
click_coordinates = []
click_count = 0
click_event = threading.Event()

def on_closing():
    # Add any cleanup or save operations here
    root.destroy()
    root.quit()

def update_progress(message):
    progress_text.config(state=tk.NORMAL)  # Enable editing the text
    progress_text.insert(tk.END, message + '\n')
    progress_text.config(state=tk.DISABLED)  # Disable editing the text
    root.update()

def clear_progress_text():
    progress_text.config(state=tk.NORMAL)  # Enable editing the text
    progress_text.delete(1.0, tk.END)  # Delete all text in the widget
    progress_text.config(state=tk.DISABLED)  # Disable editing the text

def load_image_from_explorer():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ["*.jpg", '*.png'])], initialdir='../../Dataset') #path where images will be stored
    if file_path:
        open_image(file_path)

def drop(event):
    file_path = event.data
    if file_path:
        open_image(file_path)

def handle_segmentation_mode():
    selected_mode = segmentation_mode.get()
    if selected_mode == 1:
        return 'click'
    elif selected_mode == 2:
        return 'text'

def on_image_click(event, original_image, display_image, top_level):
    global click_count

    # Get the x and y coordinates where the user clicked
    x, y = event.x, event.y
    # Calculate the ratio of resizing
    width_ratio = original_image.width / display_image.width
    height_ratio = original_image.height / display_image.height
    # Adjust the coordinates to the original image size
    original_x = int(x * width_ratio)
    original_y = int(y * height_ratio)
    click_coordinates.append((original_x, original_y))
    click_count += 1

    if click_count == 2:
        click_event.set()  # Signal that two clicks are done
        top_level.quit()
        top_level.destroy()  # Close the top-level window when two clicks are done

def get_click_coordinates(display_image, original_image):
    global click_coordinates, click_count

    # Create a top-level window for displaying the image
    top_level = tk.Toplevel()
    top_level.title("Image Coordinate Picker")

    # Create a label to display the image
    img_label = tk.Label(top_level)
    img_label.pack()

    # Convert the display image to PhotoImage
    click_image = ImageTk.PhotoImage(display_image)

    # Display the resized image
    img_label.config(image=click_image)

    # Bind the mouse click event to the image label
    img_label.bind("<Button-1>", lambda event, o=original_image, d=display_image, t=top_level: on_image_click(event, o, d, t))

    # Run the Tkinter main loop for this top-level window
    top_level.mainloop()

    # Wait for the user to click twice
    click_event.wait()
    click_event.clear()  # Reset the event

    # After the user clicks twice, adjust and return coordinates
    coords = click_coordinates[:]
    click_coordinates.clear()  # Clear the coordinates for future use
    click_count = 0  # Reset click count

    return coords

def make_merged_image(original_image, mask):
    pic = np.asarray(original_image)

    colors = ['none', 'red', 'blue']
    custom_cmap = ListedColormap(colors)

    fig_shape = pic.shape[:2]
    fig_shape = tuple(fig_shape[i]/100 for i in range(len(fig_shape)))
    fig_shape = fig_shape[::-1]

    fig = plt.figure(figsize=fig_shape,frameon=False)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(pic)
    ax.imshow(mask, cmap = custom_cmap, alpha= 0.3)
    ax.axis('off')
    ax.margins(x=0,y=0)
    fig.canvas.draw()

    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    result = Image.fromarray(result)
    return result

def open_image(file_path):
    global show_mask
    # Clear the progress text when a new image is opened
    clear_progress_text()
    
    # clear the weight output and rating
    result_label.config(text=f"Weight:")
    rating_label.config(text=f"Rating:")

    # Open the original image without resizing
    original_image = Image.open(file_path).convert('RGB')
    
    # Create a resized copy of the image for display
    display_image = original_image.copy()
    display_image.thumbnail((600, 600))  # Resize image for display

    # Convert the display image to PhotoImage
    photo = ImageTk.PhotoImage(display_image)

    # Update the label to show the resized image
    label.config(image=photo)
    label.image = photo

    # Bring the label with the image in front
    label.lift()

    # Call your functions here based on the selected segmentation mode
    selected_mode = handle_segmentation_mode()  # Get the selected mode
    
    if selected_mode == 'text':
        update_progress("Segmenting your image, please wait...")
        mask = segmentation_lang(original_image)
    elif selected_mode == 'click':
        update_progress("First click on the pot, then click on the plant...")
        coords = get_click_coordinates(display_image, original_image)
        if len(coords) == 2:  # Check if two coordinates were collected
            update_progress("Continuing with segmentation, please wait...")
            mask = segmentation_by_coordinates(coords, original_image)
        else:
            update_progress("Insufficient coordinates. Please try again.")
            return
        
    show_state = show_mask.get()
    if show_state == 1:
        segmentation_mask = mask.copy()
        mask_overlay = make_merged_image(original_image, segmentation_mask)
        mask_overlay.thumbnail((600, 600))

        new_photo = ImageTk.PhotoImage(mask_overlay)

        label.config(image=new_photo)
        label.image = new_photo

        label.lift()

    elif show_state == 0:
        # Convert the display image to PhotoImage
        photo = ImageTk.PhotoImage(display_image)

        # Update the label to show the resized image
        label.config(image=photo)
        label.image = photo

        # Bring the label with the image in front
        label.lift()
    
    update_progress("Creating dataframe...")
    dataframe = make_dataframe_from_array(mask, file_path)
    update_progress("Rating your plant's growth performance...")
    weight, rating = weight_estimation(dataframe)
    update_progress("DONE!")

    # Display the weight and rating to the user (replace this with your desired UI)
    result_label.config(text=f"Weight: {weight}g")
    rating_label.config(text=f"Rating: {rating}/5")

    label.lift()
# Create the main application window using TkinterDnD
root = tkinterDnD.Tk()
root.title("Biomass Estimator")

# Create a tkinter IntVar to hold the selected mode
segmentation_mode = tk.IntVar(value=1)
show_mask = tk.IntVar(value=1)

# set window position
#monitors = si.get_monitors()
width = 600
height = 900
screen_width = root.winfo_screenwidth()  # Width of the screen
screen_height = root.winfo_screenheight() # Height of the screen
 
# Calculate Starting X and Y coordinates for Window
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
 
root.geometry('%dx%d+%d+%d' % (width, height, x, y))

# Create a frame to hold the label with the image and the guide label
image_frame = tk.Frame(root, bg="lightgray")
image_frame.pack(padx=10, pady=10, fill="both", expand=True)  # Allow frame to expand

# Create a label to display the image with a bigger size and different background color
label = tk.Label(image_frame, bg="lightgray")
label.grid(row=0, column=0, sticky="nsew")  # Place the image label in the top-left corner

# Create a label to guide the user for drag and drop
# Enable drag and drop functionality using tkinterDnD
image_frame.drop_target_register(DND_FILES)
image_frame.dnd_bind('<<Drop>>', drop)
guide_label = tk.Label(image_frame, text="Drag and Drop Image Here.\nAlternatively you can use the \"Open Image\" button below.", fg="gray", bg="lightgray")
guide_label.grid(row=0, column=0, sticky="nsew")  # Place the guide label in the top-left corner

# Configure grid to allow the image label to expand and fill the frame
image_frame.grid_rowconfigure(0, weight=1)
image_frame.grid_columnconfigure(0, weight=1)

# Create a button to open the file dialog
button = tk.Button(root, text="Open Image", command=load_image_from_explorer)
button.pack(padx=10, pady=10)

# Create a frame to center the Radiobuttons
center_frame = tk.Frame(root)
center_frame.pack(expand=False, fill=tk.NONE)

# Create Radiobuttons for the two segmentation modes
text_prompt_mode = tk.Radiobutton(center_frame, text="By Clicking", variable=segmentation_mode, value=1, command=handle_segmentation_mode)
clicking_mode = tk.Radiobutton(center_frame, text="By Text-Prompt", variable=segmentation_mode, value=2, command=handle_segmentation_mode)

# Use the pack geometry manager to place the Radiobuttons side by side
text_prompt_mode.pack(side=tk.LEFT, pady=5)
clicking_mode.pack(side=tk.RIGHT, pady=5)

# Create a Text widget to display progress messages
progress_text = tk.Text(root, height=5, width=50)
progress_text.pack(padx=10, pady=5)
progress_text.config(state=tk.DISABLED)  # Disable editing the text

# Create labels to display the results
result_label = tk.Label(root, text="")
result_label.pack(padx=10, pady=5)

rating_label = tk.Label(root, text="")
rating_label.pack(padx=10, pady=5)

# create a frame to center the mask toggle
mask_center = tk.Frame(root)
mask_center.pack(expand=False, fill=tk.NONE)

# create a check button to toggle if the user sees the segmentation mask
toggle_mask = tk.Checkbutton(mask_center, text="Show Segmentation Mask", variable=show_mask, onvalue=1, offvalue=0)
toggle_mask.pack()

# register the window being closed
root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the application
root.mainloop()

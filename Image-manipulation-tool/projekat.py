from PIL import Image, ImageTk
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
import tkinter as tk
from tkinter import filedialog
import tkinter as ttk
from tkinter import messagebox
from ttkthemes import ThemedStyle


#-------------Bilinear interpolation----------------
def bilinear_interpolation(image, new_height, new_width):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the original image
    height, width, _ = img_array.shape

    # Calculate the scaling factors
    scale_y = new_height / height
    scale_x = new_width / width

    # Initialize the new image array
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Perform bilinear interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the coordinates in the original image
            y = i / scale_y
            x = j / scale_x

            # Get the four surrounding pixels
            y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
            y_ceil, x_ceil = min(y_floor + 1, height - 1), min(x_floor + 1, width - 1)

            # Bilinear interpolation
            top_left = img_array[y_floor, x_floor] * ((y_ceil - y) * (x_ceil - x))
            top_right = img_array[y_floor, x_ceil] * ((y_ceil - y) * (x - x_floor))
            bottom_left = img_array[y_ceil, x_floor] * ((y - y_floor) * (x_ceil - x))
            bottom_right = img_array[y_ceil, x_ceil] * ((y - y_floor) * (x - x_floor))

            # Assign the interpolated value to the new image
            new_img_array[i, j] = top_left + top_right + bottom_left + bottom_right

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_img_array)

    return new_image


#---------------Cubic interpolation for Bicubic interpolation----------------
def cubic_interp(x):
    # Cubic interpolation kernel
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3

    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

#---------------Bicubic interpolation-----------------
def bicubic_interpolation(image, new_height, new_width):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    scale_y = new_height / height
    scale_x = new_width / width
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.float32)

    for i in range(new_height):
        for j in range(new_width):
            y = i / scale_y
            x = j / scale_x
            y_vals = np.array([y + 1, y, y - 1, y - 2]) % height
            x_vals = np.array([x + 1, x, x - 1, x - 2]) % width
            cubic_weights_y = cubic_interp(y_vals - y)
            cubic_weights_x = cubic_interp(x_vals - x)

            for channel in range(3):
                patch = img_array[y_vals.astype(int), :, channel]
                patch = patch[:, x_vals.astype(int)]

                # Apply bicubic interpolation with clipping
                new_img_array[i, j, channel] = np.clip(np.sum(cubic_weights_y * patch * cubic_weights_x.T), 0, 255)

    new_image = Image.fromarray(new_img_array.astype(np.uint8))
    return new_image

#--------------- Nearest-neighbor interpolation -----------------
def nearest_neighbor_interpolation(image, new_height, new_width):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the original image
    height, width, _ = img_array.shape

    # Calculate the scaling factors
    scale_y = height / new_height
    scale_x = width / new_width

    # Initialize the new image array
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Perform nearest-neighbor interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the coordinates in the original image
            y = int(i * scale_y)
            x = int(j * scale_x)

            # Assign the nearest-neighbor pixel value to the new image
            new_img_array[i, j] = img_array[y, x]

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_img_array)

    return new_image

#--------------- B-spline interpolation -----------------

def bspline_interpolation(image, new_height, new_width):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the original image
    height, width, _ = img_array.shape

    # Create a RectBivariateSpline for each color channel
    x_vals = np.arange(0, width, 1)
    y_vals = np.arange(0, height, 1)

    # B-spline interpolation for each color channel
    bspline_r = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 0])
    bspline_g = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 1])
    bspline_b = RectBivariateSpline(y_vals, x_vals, img_array[:, :, 2])

    # Evaluate the B-spline at new coordinates
    new_y_vals = np.linspace(0, height - 1, new_height)
    new_x_vals = np.linspace(0, width - 1, new_width)

    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Evaluate B-spline for each color channel
            new_img_array[i, j, 0] = int(bspline_r(new_y_vals[i], new_x_vals[j]))
            new_img_array[i, j, 1] = int(bspline_g(new_y_vals[i], new_x_vals[j]))
            new_img_array[i, j, 2] = int(bspline_b(new_y_vals[i], new_x_vals[j]))

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_img_array)

    return new_image

#--------------- Lanczos interpolation -----------------
def lanczos_interpolation(image, new_height, new_width, a=3):
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Get the dimensions of the original image
    height, width, _ = img_array.shape

    # Calculate the scaling factors
    scale_y = height / new_height
    scale_x = width / new_width

    # Initialize the new image array
    new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Generate Lanczos interpolation function for each color channel
    x_vals = np.arange(0, width)
    y_vals = np.arange(0, height)
    interp_r = interp2d(x_vals, y_vals, img_array[:, :, 0], kind='cubic')
    interp_g = interp2d(x_vals, y_vals, img_array[:, :, 1], kind='cubic')
    interp_b = interp2d(x_vals, y_vals, img_array[:, :, 2], kind='cubic')

    # Perform Lanczos interpolation
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the coordinates in the original image
            y = i * scale_y
            x = j * scale_x

            # Evaluate Lanczos interpolation for each color channel
            new_img_array[i, j, 0] = int(interp_r(x, y))
            new_img_array[i, j, 1] = int(interp_g(x, y))
            new_img_array[i, j, 2] = int(interp_b(x, y))

    # Convert the NumPy array back to an image
    new_image = Image.fromarray(new_img_array)

    return new_image






"""
# Example usage
input_image_path = 'resources/input/liman1.jpg'
output_image_path = 'resources/output/output_image2.jpg'
new_height, new_width = 500, 700

# Open the image
input_image = Image.open(input_image_path)

# Perform bilinear interpolation
output_image = lanczos_interpolation(input_image, new_height, new_width)

# Save the result
output_image.save(output_image_path)

"""


class ImageInterpolatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Interpolator")

        # Variables
        self.input_image = None
        self.output_image = None
        self.new_height_var = tk.IntVar()
        self.new_width_var = tk.IntVar()
        self.interpolation_method_var = tk.StringVar()
        self.interpolation_methods = ["Bilinear", "Bicubic", "Nearest Neighbor", "B-spline", "Lanczos"]

        # Themed Style
        style = ThemedStyle(master)
        style.set_theme("arc")  # You can change the theme to your preference

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = ttk.Label(self.master, text="Image Interpolator", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Open Image Button
        open_button = ttk.Button(self.master, text="Open Image", command=self.open_image)
        open_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Interpolation Method Menu
        method_label = ttk.Label(self.master, text="Interpolation Method:")
        method_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        method_menu = ttk.Combobox(self.master, textvariable=self.interpolation_method_var, values=self.interpolation_methods)
        self.interpolation_method_var.set(self.interpolation_methods[0])
        method_menu.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # New Height Entry
        height_label = ttk.Label(self.master, text="New Height:")
        height_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        height_entry = ttk.Entry(self.master, textvariable=self.new_height_var)
        height_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # New Width Entry
        width_label = ttk.Label(self.master, text="New Width:")
        width_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        width_entry = ttk.Entry(self.master, textvariable=self.new_width_var)
        width_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Interpolate Button
        interpolate_button = ttk.Button(self.master, text="Interpolate", command=self.interpolate_image)
        interpolate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Display Canvas
        self.canvas = tk.Canvas(self.master, bg="white", width=800, height=600, relief="sunken", borderwidth=2)
        self.canvas.grid(row=1, column=2, rowspan=5, padx=10, pady=10, sticky="nsew")

        # Configure Grid Row/Column Weights for Resizability
        self.master.columnconfigure(2, weight=1)
        self.master.rowconfigure(1, weight=1)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            self.input_image = Image.open(file_path)
            self.display_image(self.input_image)

    def display_image(self, image):
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image

    def interpolate_image(self):
        if self.input_image is None:
            messagebox.showerror("Error", "Please open an image first.")
            return

        try:
            new_height = self.new_height_var.get()
            new_width = self.new_width_var.get()

            if new_height <= 0 or new_width <= 0:
                messagebox.showerror("Error", "Invalid dimensions. Please enter positive values.")
                return

            interpolation_method = self.interpolation_method_var.get()

            if interpolation_method == "Bilinear":
                self.output_image = self.bilinear_interpolation(new_height, new_width)
            elif interpolation_method == "Bicubic":
                self.output_image = self.bicubic_interpolation(new_height, new_width)
            elif interpolation_method == "Nearest Neighbor":
                self.output_image = self.nearest_neighbor_interpolation(new_height, new_width)
            elif interpolation_method == "B-spline":
                self.output_image = self.bspline_interpolation(new_height, new_width)
            elif interpolation_method == "Lanczos":
                self.output_image = self.lanczos_interpolation(new_height, new_width)

            self.display_image(self.output_image)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def bilinear_interpolation(self, new_height, new_width):
        img_array = np.array(self.input_image)
        img = Image.fromarray(img_array)
        resized_img = img.resize((new_width, new_height), resample=Image.BILINEAR)
        return resized_img

    def bicubic_interpolation(self, new_height, new_width):
        img_array = np.array(self.input_image)
        img = Image.fromarray(img_array)
        resized_img = img.resize((new_width, new_height), resample=Image.BICUBIC)
        return resized_img

    def nearest_neighbor_interpolation(self, new_height, new_width):
        img_array = np.array(self.input_image)
        img = Image.fromarray(img_array)
        resized_img = img.resize((new_width, new_height), resample=Image.NEAREST)
        return resized_img

    def bspline_interpolation(self, new_height, new_width):
        img_array = np.array(self.input_image)
        height, width, _ = img_array.shape
        scale_y = height / new_height
        scale_x = width / new_width
        new_img_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for channel in range(3):
            bspline = interp2d(np.arange(0, width), np.arange(0, height), img_array[:, :, channel], kind='cubic')
            new_x_vals = np.linspace(0, width - 1, new_width)
            new_y_vals = np.linspace(0, height - 1, new_height)

            for i in range(new_height):
                for j in range(new_width):
                    new_img_array[i, j, channel] = int(bspline(new_x_vals[j], new_y_vals[i]))

        return Image.fromarray(new_img_array)

    def lanczos_interpolation(self, new_height, new_width):
        img_array = np.array(self.input_image)
        img = Image.fromarray(img_array)
        resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        return resized_img

def main():
    root = tk.Tk()
    app = ImageInterpolatorApp(root)
    root.geometry("1000x800")  # Set the initial size of the window
    root.mainloop()

if __name__ == "__main__":
    main()

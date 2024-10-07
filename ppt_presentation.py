import os

from PIL import Image


def create_image_grid(
    source_directory, target_directory, grid_filename="image_grid.png", spacing=10
):
    # Ensure the target directory exists, create if not
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")

    # List all PNG files in the source directory
    png_files = [f for f in os.listdir(source_directory) if f.endswith(".png")]

    # Pick the first 9 PNG files
    png_files = png_files[:9]

    # Ensure there are exactly 9 images
    if len(png_files) < 9:
        print(grid_filename)
        raise ValueError("There are less than 9 PNG files in the source directory.")

    # Determine the size of the grid
    images = [Image.open(os.path.join(source_directory, file)) for file in png_files]
    widths, heights = zip(*(i.size for i in images))

    # Assuming all images are of the same size, create a grid
    max_width = max(widths)
    max_height = max(heights)
    grid_width = max_width * 3 + spacing * 2
    grid_height = max_height * 3 + spacing * 2
    grid_image = Image.new(
        "RGB", (grid_width, grid_height), "white"
    )  # 'white' background

    # Paste the images into the grid with spacing
    for index, image in enumerate(images):
        x = index % 3 * (max_width + spacing)
        y = index // 3 * (max_height + spacing)
        grid_image.paste(image, (x, y))

    # Save the grid image to the target directory
    grid_image_path = os.path.join(target_directory, grid_filename)
    grid_image.save(grid_image_path)
    print(f"Grid image saved to {grid_image_path}")


# Example usage
neuron_idxs = [
    1149,
    133,
    153,
    182,
    222,
    242,
    10057,
    41781,
    879,
    5254,
    30021,
    22552,
    14061,
    9311,
    13626,
    9673,
    3156,
    14704,
    1683,
    26449,
    16379,
    52097,
    4033,
    774,
    2681,
    1773,
    886,
    983,
]
target_directory = "MATS_presentaion"
for neuron_idx in neuron_idxs:
    source_directory = f"dashboard/{neuron_idx}/max_activating"
    create_image_grid(
        source_directory, target_directory, grid_filename=f"{neuron_idx}.png"
    )

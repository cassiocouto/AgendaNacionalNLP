from PIL import Image
import numpy as np

def get_unique_pixel_values(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Get unique pixel values
    unique_pixels = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
    
    return unique_pixels

def resize_mask(image_path, scale_factor):
    """
    Resize the mask image by a given scale factor.

    Parameters:
    - image_path: str, path to the mask image.
    - scale_factor: float, factor by which to scale the image dimensions.

    Returns:
    - np.array, resized mask.
    """
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    resized_image = image.resize(new_size, Image.LANCZOS)
    return np.array(resized_image)

def replace_pixel_values(original_image_path, new_image_path):
    # Open the original image
    image = Image.open(original_image_path).convert("RGBA")
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Create a copy of the image array to modify
    modified_image_array = image_array.copy()
    # Set all pixels with alpha == 0 to white (255, 255, 255, 255)
    white = [255, 255, 255, 255]
    black = [0, 0, 0, 255]
    
    # Create a mask for pixels with alpha == 0
    alpha_zero_mask = image_array[:, :, 3] == 0
    
    # Set pixels with alpha == 0 to white
    modified_image_array[alpha_zero_mask] = white
    
    # Set all other pixels to black
    modified_image_array[~alpha_zero_mask] = black
    
    # Convert the modified numpy array back to an image
    modified_image = Image.fromarray(modified_image_array.astype(np.uint8), 'RGBA')
    
    # Save the new image
    modified_image.save(new_image_path)
    
pixels = get_unique_pixel_values("assets/norte.png")
print(pixels)
replace_pixel_values("assets/norte.png", "assets/norte_modified.png")
replace_pixel_values("assets/nordeste.png", "assets/nordeste_modified.png")
replace_pixel_values("assets/centro-oeste.png", "assets/centro-oeste_modified.png")
replace_pixel_values("assets/sudeste.png", "assets/sudeste_modified.png")
replace_pixel_values("assets/sul.png", "assets/sul_modified.png")
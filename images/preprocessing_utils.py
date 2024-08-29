from PIL import Image, ExifTags

# Function to suppress rotation within an image
def suppress_rotation(image: Image.Image) -> Image.Image:
    """
    Suppresses rotation in the given image by correcting its orientation
    using EXIF metadata if available.
    ---------------------------------------------------
    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The image with suppressed rotation.
    """
    try:
        # Get the EXIF data from the image
        exif = image._getexif()

        if exif is not None:
            # Find the orientation key in EXIF data
            orientation = None
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'Orientation':
                    orientation = value
                    break


            # Rotate image according to the orientation value
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error: {e}")
        pass

    return image
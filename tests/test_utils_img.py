# coding: utf-8
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pix2text.utils import convert_transparent_to_contrasting


def create_rgba_image(width=100, height=100, bg_color=(255, 0, 0, 0), fg_color=(0, 0, 255, 255)):
    """Create a test RGBA image with transparent background and some foreground content."""
    # Create a fully transparent image
    img = Image.new('RGBA', (width, height), bg_color)
    
    # Add some non-transparent content in center
    pixels = img.load()
    for i in range(width//4, 3*width//4):
        for j in range(height//4, 3*height//4):
            pixels[i, j] = fg_color
            
    return img


def create_la_image(width=100, height=100, bg_value=0, fg_value=255):
    """Create a test LA (grayscale with alpha) image."""
    # Create a fully transparent image
    img = Image.new('LA', (width, height), (bg_value, 0))
    
    # Add some non-transparent content in center
    pixels = img.load()
    for i in range(width//4, 3*width//4):
        for j in range(height//4, 3*height//4):
            pixels[i, j] = (fg_value, 255)
            
    return img


def create_p_image_with_transparency(width=100, height=100):
    """Create a test palette mode (P) image with transparency."""
    # Start with an RGBA image
    rgba = create_rgba_image(width, height)
    
    # Convert to palette mode with transparency
    p_img = rgba.convert('P')
    
    # Set transparency
    p_img.info['transparency'] = 0
    
    return p_img


def create_rgb_image(width=100, height=100, color=(100, 150, 200)):
    """Create a test RGB image."""
    return Image.new('RGB', (width, height), color)


def test_convert_rgba_transparent():
    """Test converting an RGBA image with transparency."""
    # Create test image
    img = create_rgba_image()
    
    # Apply the function
    result = convert_transparent_to_contrasting(img)
    
    # Verify the result
    assert result.mode == 'RGB', "Result should be in RGB mode"
    assert result.size == img.size, "Image dimensions should not change"
    
    # Convert to numpy array to check pixel values
    result_array = np.array(result)
    
    # The background (originally transparent) should now have a contrasting color
    # to the blue foreground we set in create_rgba_image
    bg_sample = result_array[5, 5]  # Sample from corner (background)
    fg_sample = result_array[50, 50]  # Sample from center (foreground)
    
    # Make sure background and foreground are different
    assert not np.array_equal(bg_sample, fg_sample), "Background should have different color than foreground"


def test_convert_la_transparent():
    """Test converting an LA (grayscale with alpha) image with transparency."""
    img = create_la_image()
    result = convert_transparent_to_contrasting(img)
    
    assert result.mode == 'RGB', "Result should be in RGB mode"
    assert result.size == img.size, "Image dimensions should not change"


def test_convert_p_with_transparency():
    """Test converting a palette image with transparency."""
    img = create_p_image_with_transparency()
    result = convert_transparent_to_contrasting(img)
    
    assert result.mode == 'RGB', "Result should be in RGB mode"
    assert result.size == img.size, "Image dimensions should not change"


def test_convert_rgb_no_transparency():
    """Test converting an RGB image (no transparency)."""
    # For RGB images, we just expect a converted copy
    img = create_rgb_image()
    result = convert_transparent_to_contrasting(img)
    
    assert result.mode == 'RGB', "Result should be in RGB mode"
    assert result.size == img.size, "Image dimensions should not change"
    
    # The image should look the same as input (just ensured to be RGB)
    img_rgb = img.convert('RGB')
    assert np.array_equal(np.array(result), np.array(img_rgb)), "RGB image should not change visually"


def test_end_to_end():
    """
    Test the full workflow: create image, save it, read it, convert it,
    then check the result matches expectations.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save a test image
        test_path = os.path.join(tmp_dir, "test_transparent.png")
        img = create_rgba_image()
        img.save(test_path)
        
        # Read the image back and convert it
        img_reopened = Image.open(test_path)
        result = convert_transparent_to_contrasting(img_reopened)
        
        # Verify results
        assert result.mode == 'RGB', "Result should be in RGB mode"
        assert result.size == img.size, "Image dimensions should not change"
        
        # Save the result for comparison (optional)
        result_path = os.path.join(tmp_dir, "test_result.jpg")
        result.save(result_path)


def test_edge_cases():
    """Test edge cases like extremely small images or unusual color patterns."""
    # Test a 1x1 pixel transparent image
    tiny_image = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
    result = convert_transparent_to_contrasting(tiny_image)
    assert result.mode == 'RGB', "Result should be in RGB mode"
    assert result.size == (1, 1), "Image dimensions should not change"
    
    # Test a fully transparent image with no content
    empty_image = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
    result = convert_transparent_to_contrasting(empty_image)
    assert result.mode == 'RGB', "Result should be in RGB mode"
    
    # Test an image with partially transparent pixels
    partial_img = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
    pixels = partial_img.load()
    for i in range(50):
        for j in range(50):
            pixels[i, j] = (255, 0, 0, i % 255)  # Varying alpha values
    result = convert_transparent_to_contrasting(partial_img)
    assert result.mode == 'RGB', "Result should be in RGB mode"


if __name__ == "__main__":
    # Run tests manually if needed
    test_convert_rgba_transparent()
    test_convert_la_transparent()
    test_convert_p_with_transparency()
    test_convert_rgb_no_transparency()
    test_end_to_end()
    test_edge_cases()
    print("All tests passed!")

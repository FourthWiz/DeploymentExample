"""
This module contains a function that moves a file from the source to the destination and extracts it.
"""

import os
import zipfile


def move(source, destination):
    """
    This function moves a file from the source to the destination.
    """
    # Move the file
    os.rename(source, destination)

def move_extract(source, destination):
    """
    This function moves a file from the source to the destination and extracts it.
    """
    # Move the file
    os.rename(source, os.path.join(destination, source))
    # Extract the file
    with zipfile.ZipFile(os.path.join(destination, source), 'r') as zip_ref:
        zip_ref.extractall(destination)
    # Remove the zip file
    os.remove(os.path.join(destination, source))

def extract(source, file_name):
    """
    This function moves a file from the source to the destination and extracts it.
    """
    # Extract the file
    with zipfile.ZipFile(os.path.join(source, file_name), 'r') as zip_ref:
        zip_ref.extractall(source)
    # Remove the zip file
    os.remove(os.path.join(source, file_name))
"""Contains functions to detect object types, manipulate objects, etc."""

from datetime import datetime

def generate_clip_name():
    now = datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S') + f",{now.microsecond // 1000}"
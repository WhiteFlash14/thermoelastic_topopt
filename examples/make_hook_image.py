"""Draws examples/hook.png - a black-on-white picture used by hook_from_image.yaml.

Any picture works the same way: draw your part in black on a white background
(Paint, Inkscape export, a scanned sketch ...), save as PNG and point a
`{type: image, file: ..., width: ...}` shape at it.
"""
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 300, 240                      # pixels; the YAML maps 300 px -> 150 mm
img = Image.new('L', (W, H), 255)
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 80, 240], fill=0)                 # vertical post (left)
d.rectangle([0, 0, 300, 60], fill=0)                 # top arm
d.pieslice([170, 0, 300, 160], 270, 90, fill=0)      # rounded end of the arm
d.rectangle([220, 60, 300, 150], fill=0)             # hanging tip
d.ellipse([235, 90, 275, 130], fill=255)             # hole in the tip (load pin)
Path(__file__).with_name('hook.png').parent.mkdir(exist_ok=True)
img.save(Path(__file__).with_name('hook.png'))
print('wrote', Path(__file__).with_name('hook.png'))

"""Scene manipulation"""

import bpy

# Get the object camera
obj = bpy.data.objects["Camera"]

print(obj.matrix_world.to_3x3())
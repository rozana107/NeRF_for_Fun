"""Scene manipulation in Blender script"""

import bpy
import math
from mathutils import Vector

# === Scene Settings ===
OBJECT_SIZE = 1.0
OBJECT_LOCATION = (0, 0, 0.5)

CAMERA_SPHERE_RADIUS = 5.0
CAMERA_IMAGE_SIZE = (200, 200)
IMAGE_FILEPATH = "D:/My_Computer/Work projects/Research_Coding_Interview/Blender/images/"

THETA_STEPS = 10  # number of elevation rows (from top to equator)
PHI_STEPS_BASE = 16  # base azimuth steps for the first theta row (more added as theta increases)

# === Clear Existing Objects ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === Add Monkey Object ===
bpy.ops.mesh.primitive_monkey_add(size=OBJECT_SIZE, location=OBJECT_LOCATION)

# === Get or Add Camera ===
camera = bpy.data.objects.get("Camera")
if not camera:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
bpy.context.scene.camera = camera

# === Add Point Light ===
light_data = bpy.data.lights.new(name="Point_Light", type='POINT')
light_data.energy = 1000  # Adjust the brightness as needed
light_object = bpy.data.objects.new(name="Point_Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)

# Position the light slightly outside the camera sphere, above the object
light_object.location = (CAMERA_SPHERE_RADIUS, CAMERA_SPHERE_RADIUS, CAMERA_SPHERE_RADIUS + 2)

# === Set Render Resolution ===
bpy.context.scene.render.resolution_x = CAMERA_IMAGE_SIZE[0]
bpy.context.scene.render.resolution_y = CAMERA_IMAGE_SIZE[1]

# === Generate Hemisphere Grid Samples ===
sample_index = 0
for i in range(THETA_STEPS):
    # θ from 0 (top) to π/2 (equator)
    theta = (math.pi / 2) * (i + 1) / (THETA_STEPS + 1)

    # More phi samples as theta increases (use sin(theta) for density)
    phi_steps = max(4, round(PHI_STEPS_BASE * math.sin(theta)))
    for j in range(phi_steps):
        phi = 2 * math.pi * j / phi_steps

        # Spherical to Cartesian
        x = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.cos(phi)
        y = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.sin(phi)
        z = CAMERA_SPHERE_RADIUS * math.cos(theta)

        # Move and rotate camera
        camera.location = (x, y, z)
        direction = Vector(OBJECT_LOCATION) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

        # Optional: add a debug sphere at camera position
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.05, location=(x, y, z))

        # Render
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True, use_viewport=True)
        bpy.data.images['Render Result'].save_render(
            filepath=f"{IMAGE_FILEPATH}render_{sample_index:03d}_theta_{theta:.2f}_phi_{phi:.2f}.png"
        )

        sample_index += 1

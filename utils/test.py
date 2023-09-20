import matplotlib.pyplot as plt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Draw lens (as a vertical line)
ax.plot([0, 0], [-2, 2], 'b-', linewidth=6, label='Lens')

# Draw image plane (as a vertical line)
ax.plot([2, 2], [-2, 2], 'g--', linewidth=2, label='Image Plane (Sensor)')

# Draw rays of light (parallel to the principal axis)
for y in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
    ax.plot([-2, 2], [y, y], 'r-', alpha=0.5)
    ax.plot([-2, 2], [y, -y], 'r-', alpha=0.5)

# Mark the focal point
ax.plot(1, 0, 'go', markersize=10, label='Focal Point')

# Mark the optical center (principal point)
ax.plot(2, 0, 'mo', markersize=10, label='Optical Center (Principal Point)')

# Annotations and styling
ax.text(-0.5, 1.8, 'Incoming Light Rays', color='red', fontsize=10)
ax.text(2.2, 0.1, '(c_x, c_y)', color='magenta', fontsize=9)
ax.set_xlim(-2.5, 3)
ax.set_ylim(-2, 2)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_aspect('equal', 'box')
ax.axis('off')
ax.legend(loc='upper right')

plt.title("Camera Optics Visualization")
plt.show()

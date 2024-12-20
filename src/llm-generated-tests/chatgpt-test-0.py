import matplotlib.pyplot as plt

# Define the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Display the mathematical notation manually
ax.text(0.5, 0.8, r'$\mathbf{x}$', fontsize=24, ha='center', va='center')
ax.text(0.5, 0.6, r'$\mathbf{W}$', fontsize=24, ha='center', va='center')
ax.text(0.5, 0.4, r'$\mathbf{x} \cdot \mathbf{W}$', fontsize=24, ha='center', va='center')
ax.text(0.5, 0.2, r'$\mathbf{x} \cdot \mathbf{W} + b$', fontsize=24, ha='center', va='center')
ax.text(0.5, 0.0, r'$\sigma(\mathbf{x} \cdot \mathbf{W} + b)$', fontsize=24, ha='center', va='center')

# Show the plot
plt.axis('off')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Utils
def compute_euclidean_distances(X, point):
    distances = np.linalg.norm(X - point, axis=1)
    return distances

# def plot_tetrahedron(vertices, gt, est=None, size=2):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Create list of faces from the vertices
#     faces = [
#         [vertices[0], vertices[1], vertices[2]],
#         [vertices[0], vertices[1], vertices[3]],
#         [vertices[0], vertices[2], vertices[3]],
#         [vertices[1], vertices[2], vertices[3]]
#     ]
    
#     # Plot the faces
#     poly3d = Poly3DCollection(faces, alpha=0.5, linewidths=1, edgecolors='r')
#     ax.add_collection3d(poly3d)
    
#     # Scatter plot of the vertices
#     ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b', s=100)
#     ax.scatter(gt[0], gt[1], gt[2], color='r', s=100)
#     if est is not None:
#         ax.scatter(est[0], est[1], est[2], color='g', s=100)

#     # Set plot limits
#     ax.set_xlim([-size, size])
#     ax.set_ylim([-size, size])
#     ax.set_zlim([-size, size])

#     # Labels for axes
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     plt.show()

def plot_points_2d(data):
    X = data['X']
    point = data['node_pos']

    """
    Plots points in 2D space based on the input matrix X.

    Parameters:
    X (numpy.ndarray): A matrix of shape (n, 2) where each row represents a point's (x, y) coordinates in 2D space.
    """
    if X.shape[1] != 2:
        raise ValueError("Input matrix X must have exactly 2 columns for 2D points.")
    
    # Extract x and y coordinates
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    
    # Create a scatter plot
    plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Anchors')
    plt.scatter(point[0], point[1], color='red', marker='^', label='Tag')

    # Label the axes
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.axis('equal')
    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
    
def plot_robot_setup(data):
    X = data['anchors']
    point = data['x0']

    # Print trajectory
    trajectory = data['trajectory_gt']
    plt.plot(trajectory[:, 0], trajectory[:, 1], '--g', label='Ground truth trajectory')

    # Extract x and y coordinates
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    
    # Create a scatter plot
    plt.scatter(x_coords, y_coords, color='blue', marker='o', label='Anchors')
    plt.scatter(point[0], point[1], color='red', marker='^', label='Robot')

    # Label the axes
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    plt.axis('equal')
    # Add grid and legend
    plt.grid(True)
    plt.legend()


    # Show the plot
    plt.show()

def plot_estimated_trajectory(data, x_plot):
	plt.plot(x_plot[:, 0],x_plot[:, 1], label='estimated')
	trajectory = data['trajectory_gt']
	plt.plot(trajectory[:, 0], trajectory[:, 1], label='ground truth')
	plt.legend()
	plt.show()

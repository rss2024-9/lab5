import numpy as np

class MotionModel:

    def __init__(self, node):
        self.node = node
        self.deterministic = False

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """        
        # Creates a rotation + translation matrix
        T = lambda x: np.array([
            [+np.cos(x[2]), -np.sin(x[2]), x[0]],
            [+np.sin(x[2]), +np.cos(x[2]), x[1]],
            [0,             0,             1   ],
        ])

        dT = T(odometry)
        for i, particle in enumerate(particles):
            t = np.matmul(T(particle), dT)
            particles[i, 0] = t[0, 2]
            particles[i, 1] = t[1, 2]
            particles[i, 2] = np.arctan2(t[1, 0], t[0, 0])

        # Noise +/- range for (x, y, theta)
        noise = (0.05, 0.05, 0.15)
        particles += (np.random.random(particles.shape) - 0.5) * 2 * noise

        return particles
 

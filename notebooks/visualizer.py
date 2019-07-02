# external imports
import time
from numpy import pi
from meshcat import Visualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# internal imports
from dynamics import l, d, h

# initialize visualizer
vis = Visualizer()

# cart-pole
cart_pole = vis['cart_pole']

# cart
cart = cart_pole['cart']
cart.set_object(g.Box([.3*l, .3*l, .3*l]), g.MeshLambertMaterial(color=0xff2222))

# pivot
pivot = cart['pivot']
pivot.set_transform(tf.rotation_matrix(pi/2, [1, 0., 0.]))

# pole
pole = pivot['pole']
pole.set_object(g.Box([.05*l, l, .05*l]), g.MeshLambertMaterial(color=0x2222ff))
pole.set_transform(tf.translation_matrix([0., .5, 0.]))

# left wall
left_wall = vis['left_wall']
left_wall.set_object(g.Box([l, .05*l, l]), g.MeshLambertMaterial(color=0x22ff22))
left_wall.set_transform(tf.translation_matrix([0., -d, l]))

# right wall
right_wall = vis['right_wall']
right_wall.set_object(g.Box([l, .05*l, l]), g.MeshLambertMaterial(color=0x22ff22))
right_wall.set_transform(tf.translation_matrix([0., d, l]))

# animation
def visualize(x):
    cart.set_transform(tf.translation_matrix([0, x[0], 0]))
    pivot.set_transform(tf.rotation_matrix(x[1] + pi/2, [1, 0, 0]))
def animate(x_list):
	for xt in x_list:
	    visualize(xt)
	    time.sleep(h)
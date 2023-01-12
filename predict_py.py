import pygame
import numpy as np
import pickle
from nn_class_py import Layer, Model

with open('300-100-10-softmax-e300.pkl', 'rb') as p:
    model = pickle.load(p)

# Initialize Pygame
pygame.init()

# Set the window size
screen_width, screen_height = 800, 560
screen = pygame.display.set_mode((screen_width, screen_height))

# Create a 28x28 numpy array to store the image
image = np.zeros((28, 28), dtype=np.uint8)

# Scale the image to the size of the window
image_scale = 20

# Create the label font
font = pygame.font.Font(None, 24)

# Create the label variable
label = "Prediction: None"
brush_size = 1
# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for mouse button down events
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Get the mouse position
            mouse_x, mouse_y = event.pos
            # Scale the mouse position to match the image size
            image_x = mouse_x // image_scale
            image_y = mouse_y // image_scale
            # Set the pixel at the mouse position to black
            if (image_y < 28 and image_x < 28):
                if event.button == 1:
                    # image[max()]
                    image[image_y-brush_size:image_y+brush_size, image_x-brush_size:image_x+brush_size] = 255
                elif event.button == 3:
                    image[image_y, image_x] = 0
                img = image.reshape(1,784).T
                label = "Prediction: " + str(model.make_predictions(img)[0])
                if event.button == 2:
                    screen.fill((255, 255, 255))
                    image = np.zeros((28, 28), dtype=np.uint8)
                    label = "Prediction: " + 'None'

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the image to the screen
    for y in range(28):
        for x in range(28):
            color = (image[y, x], image[y, x], image[y, x])
            rect = pygame.Rect(x * image_scale, y * image_scale, image_scale, image_scale)
            pygame.draw.rect(screen, color, rect)

    # Update the screen
    label_image = font.render(label, True, (0, 0, 0))
    screen.blit(label_image, (580, 250))
    pygame.display.update()

# Quit Pygame
pygame.quit()
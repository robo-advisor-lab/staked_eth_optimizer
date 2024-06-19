import cairo

# Print the cairo version
print("Cairo version:", cairo.version)
print("Cairo version string:", cairo.cairo_version_string())

# Create a simple image to test the library
WIDTH, HEIGHT = 256, 256

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
context = cairo.Context(surface)

context.set_source_rgb(0, 0, 0)
context.paint()

context.set_source_rgb(1, 1, 1)
context.move_to(128, 128)
context.arc(128, 128, 100, 0, 2 * 3.14)
context.stroke()

surface.write_to_png("output.png")
print("Test image 'output.png' created successfully.")

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

contract_address = os.getenv('CONTRACT_ADDRESS')
private_key = os.getenv('PRIVATE_KEY')

print(f'Contract Address: {contract_address}')
print(f'Private Key: {private_key}')

import os
import sys

from builderHigh import create_builder


if __name__ == "__main__":

    # Create our world builder
    builder = create_builder()

    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our
    # own media resource folder with MATRX.
    media_folder = os.path.dirname(os.path.join(os.path.realpath("/home/ruben/Documents/MATRX/MATRX"), "media"))
    builder.startup(media_folder=media_folder)

    for world in builder.worlds(nr_of_worlds=1):
        print("Started world...")
        world.run(builder.api_info)
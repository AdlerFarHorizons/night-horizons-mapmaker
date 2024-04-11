#!/bin/bash

# Call pyreverse to generate
pyreverse -o html -k night_horizons

# Rename
mv classes.html gen_classes.html
mv packages.html gen_packages.html
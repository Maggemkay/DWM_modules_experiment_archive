#!/bin/bash

# Remove old plots
rm -r out 2>/dev/null

file="plot_addmul_performance.py"

# Run one plot
echo "Running $file"
/usr/bin/env python3 $file
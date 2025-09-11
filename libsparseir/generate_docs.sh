#!/bin/bash

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen is not installed. Please install Doxygen first."
    echo "On macOS, you can install it using: brew install doxygen"
    exit 1
fi

# Check if Graphviz is installed (for class diagrams)
if ! command -v dot &> /dev/null; then
    echo "Warning: Graphviz is not installed. Class diagrams will not be generated."
    echo "On macOS, you can install it using: brew install graphviz"
fi

# Create docs directory if it doesn't exist
mkdir -p docs

# Generate documentation
echo "Generating documentation..."
doxygen Doxyfile

if [ $? -eq 0 ]; then
    echo "Documentation generated successfully in docs/html directory."
    echo "You can open docs/html/index.html in your web browser."
else
    echo "Error: Documentation generation failed."
    exit 1
fi
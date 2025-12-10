import os

# Base directories
base_dirs = {
    "include": {
        "layers": ["Layer.h", "Dense.h", "Activation.h"],
        "losses": ["Loss.h", "MSELoss.h", "CrossEntropyLoss.h"],
        "optimizers": ["Optimizer.h", "SGD.h", "Adam.h"],
        "data": ["Dataset.h"],
        "utils": ["Utils.h", "Tensor.h"]
    },
    "src": {
        "layers": ["Dense.cpp", "Activation.cpp"],
        "losses": ["MSELoss.cpp", "CrossEntropyLoss.cpp"],
        "optimizers": ["SGD.cpp", "Adam.cpp"],
        "data": ["Dataset.cpp"],
        "utils": ["Utils.cpp", "Tensor.cpp"],
        "": ["main.cpp"]  # Root src folder
    }
}

# Comments for each file
comments = {
    "Tensor.h": "// Tensor.h / Tensor.cpp\n// Defines Tensor class for multi-dimensional arrays.\n// Supports basic math operations and broadcasting.\n",
    "Tensor.cpp": "// Tensor.cpp\n// Implementation of Tensor class methods.\n",
    "Math.h": "// Math.h / Math.cpp\n// Helper math functions: matrix multiply, dot product, transpose, random init.\n",
    "Math.cpp": "// Math.cpp\n// Implementation of math helper functions.\n",
    "Layer.h": "// Layer.h\n// Abstract base class for all neural network layers.\n// Requires forward() and backward() methods.\n",
    "Dense.h": "// Dense.h / Dense.cpp\n// Fully connected layer. Implements forward and backward pass.\n",
    "Dense.cpp": "// Dense.cpp\n// Implementation of Dense layer methods.\n",
    "Activation.h": "// Activation.h / Activation.cpp\n// Activation functions: ReLU, Sigmoid, Tanh.\n",
    "Activation.cpp": "// Activation.cpp\n// Implementation of activation functions.\n",
    "Loss.h": "// Loss.h\n// Base class for loss functions. forward() and backward() methods.\n",
    "MSELoss.h": "// MSELoss.h / MSELoss.cpp\n// Mean Squared Error loss.\n",
    "MSELoss.cpp": "// MSELoss.cpp\n// Implementation of MSE loss.\n",
    "CrossEntropyLoss.h": "// CrossEntropyLoss.h / CrossEntropyLoss.cpp\n// Cross-Entropy loss for classification.\n",
    "CrossEntropyLoss.cpp": "// CrossEntropyLoss.cpp\n// Implementation of Cross-Entropy loss.\n",
    "Optimizer.h": "// Optimizer.h\n// Base class for optimizers. update() method required.\n",
    "SGD.h": "// SGD.h / SGD.cpp\n// Stochastic Gradient Descent optimizer.\n",
    "SGD.cpp": "// SGD.cpp\n// Implementation of SGD.\n",
    "Adam.h": "// Adam.h / Adam.cpp\n// Adam optimizer.\n",
    "Adam.cpp": "// Adam.cpp\n// Implementation of Adam optimizer.\n",
    "Dataset.h": "// Dataset.h / Dataset.cpp\n// Handles dataset loading, batching, and shuffling.\n",
    "Dataset.cpp": "// Dataset.cpp\n// Implementation of Dataset methods.\n",
    "Utils.h": "// Utils.h / Utils.cpp\n// Helper functions: metrics, printing, etc.\n",
    "Utils.cpp": "// Utils.cpp\n// Implementation of utility functions.\n",
    "main.cpp": "// main.cpp\n// Example usage of ML library.\n#include \"Tensor.h\"\n#include \"layers/Dense.h\"\n#include \"layers/Activation.h\"\n#include \"losses/Loss.h\"\n#include \"optimizers/Optimizer.h\"\n#include \"Model.h\"\n\nint main() {\n    // TODO: Implement ML model usage here\n    return 0;\n}\n"
}

# Create folders and files
for base, subdirs in base_dirs.items():
    for subdir, files in subdirs.items():
        folder_path = os.path.join(base, subdir) if subdir else base
        os.makedirs(folder_path, exist_ok=True)
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            content = comments.get(file_name, f"// {file_name}\n// TODO: implement this file\n")
            with open(file_path, "w") as f:
                f.write(content)

print("ML library folder and file structure created successfully!")







# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ManimML is a Python library that provides animations and visualizations of machine learning concepts using the [Manim Community Library](https://www.manim.community/). It focuses on creating reusable, composable visualization primitives that can be combined to explain complex ML architectures.

**Key Publication**: [Paper on arXiv](https://arxiv.org/abs/2306.17108)

## Build, Test, and Lint Commands

### Setup

The project uses setuptools. Install locally with:
```bash
pip install -e .
```

### Linting and Code Style

Black formatter and pydocstyle are used for code quality:
```bash
# Format code with Black
black .

# Check code style
pydocstyle .

# Combined style check (as defined in Makefile)
make checkstyle
```

CI runs a Black linter via GitHub Actions (`.github/workflows/black.yml`) on every push and pull request.

### Running Tests

Tests use pytest with custom graphical frame comparison testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_feed_forward.py

# Run specific test
pytest tests/test_feed_forward.py::test_FeedForwardScene

# Skip slow tests (marked with @pytest.mark.slow)
pytest --skip_slow

# Show visual diff for failed graphical tests
pytest --show_diff

# Generate control frames for graphical tests (creates reference data)
pytest --set_test
```

**Important**: Graphical unit tests require control frames (reference data) stored in `tests/control_data/`. Use `--set_test` flag to generate them for new tests.

Test Configuration: `tests/conftest.py` provides pytest fixtures and CLI options for graphical testing with Manim.

### Publishing

To publish to PyPI:
```bash
python3 -m build
python3 -m twine upload --repository pypi dist/*
```

## Architecture Overview

### Core Design Principles

ManimML uses a **layered, composable architecture** where visualization components connect automatically based on type matching. The system creates forward-pass animations by connecting typed layers into a neural network graph.

### Main Components

#### 1. Neural Network Container (manim_ml/neural_network/)

**NeuralNetwork class** - The main orchestrator:
- Takes a list of input layers (e.g., `[FeedForwardLayer(3), ConvLayer(5, 3, 3)]`)
- Automatically constructs connective layers between incompatible adjacent layers
- Manages layout (currently linear left-to-right)
- Provides forward-pass animation generation
- Extends Manim's `Group` mobject

Key methods:
- `make_forward_pass_animation()` - Creates animation of data flowing through the network
- `add_connection()` - Manually connect layers
- `remove_layer()` / `insert_layer()` - Dynamic network modification

#### 2. Layer Hierarchy

**Parent Classes** (manim_ml/neural_network/layers/parent_layers.py):
- `NeuralNetworkLayer` - Abstract base for all layer types
- `VGroupNeuralNetworkLayer` - Base for VGroup-based layers
- `ConnectiveLayer` - Base for connecting two layers (handles animation between them)
- `ThreeDLayer` - Mixin for 3D layer support

**Core Layer Types**:
- `FeedForwardLayer` - Traditional fully-connected layer with circular nodes
- `Convolutional2DLayer` - 2D convolution with 3D visualization of filters and feature maps
- `ImageLayer` - Input image layer with homotopy animation
- `MaxPooling2DLayer` - Max pooling operation visualization
- `EmbeddingLayer` - Embedding space visualization
- `VectorLayer`, `PairedQueryLayer`, `TripletLayer` - Specialized layer types

**Connective Layers** - Automatically created between incompatible layer pairs:
- Named pattern: `[InputType]To[OutputType]` (e.g., `FeedForwardToFeedForward`, `ConvolutionalToFeedForward`)
- Located in `manim_ml/neural_network/layers/`
- Auto-discovery via `get_connective_layer()` in `util.py`

#### 3. Layer Construction and Animation Flow

All layers follow a two-phase pattern:

**Phase 1: Construction** - `construct_layer(input_layer, output_layer, **kwargs)`
- Called during NeuralNetwork initialization
- Builds Manim mobjects (circles, rectangles, etc.)
- Sets up geometry and visual styling

**Phase 2: Animation** - `make_forward_pass_animation(layer_args={}, **kwargs)`
- Returns Manim Animation objects
- Visualizes data flow through the layer
- Called when user requests forward-pass animation

#### 4. Animation System

**Forward Pass Animations**:
- `NeuralNetwork.make_forward_pass_animation()` composes animations from all layers
- Each layer produces its own animation
- Connective layers animate the data flow/connections between layers
- Animations can be sequential or parallel via `AnimationGroup`

**Dynamic Transformations** (manim_ml/neural_network/animations/):
- `RemoveLayer` / `InsertLayer` - Dynamically modify network topology
- `dropout.py` - Special dropout animation

#### 5. Activation Functions

Located in `manim_ml/neural_network/activation_functions/`:
- Abstract `ActivationFunction` base class
- Implementations: `ReLU`, `Sigmoid`
- Layers accept `activation_function` parameter (string or instance)
- Visualized as graphs above the layer

#### 6. Configuration and Styling

**Color Schemes** (manim_ml/utils/colorschemes/colorschemes.py):
- `dark_mode` (default) - Black background, white/orange accents
- `light_mode` - White background, blue accents
- Used globally via `manim_ml.config.color_scheme`

**Global Config** (manim_ml/__init__.py):
- `ManimMLConfig` class manages color scheme and 3D rotation settings
- Accessible via `manim_ml.config`
- 3D scenes use default rotation angles (90 degrees around X-axis)

#### 7. Utility Modules

**Mobjects** (manim_ml/utils/mobjects/):
- `connections.py` - `NetworkConnection` for animating edges
- `gridded_rectangle.py` - Rectangle variants for layer styling
- `list_group.py` - `ListGroup` - enhanced VGroup with indexing
- `plotting.py` - Plot visualization utilities
- `image.py` - Image handling and manipulation

**Testing** (manim_ml/utils/testing/):
- `frames_comparison.py` - `@frames_comparison` decorator for graphical unit tests
- Stores control frames in `control_data/[module_name]/[test_name].npz`
- Supports `--set_test` for generating baseline frames

#### 8. Additional Modules

**Decision Trees** (manim_ml/decision_tree/):
- `DecisionTreeDiagram` - Visualizes sklearn decision trees
- `SplitNode` / `LeafNode` - Node types

**Diffusion** (manim_ml/diffusion/):
- `mcmc.py` - MCMC visualization helpers

**Architectures** (manim_ml/neural_network/architectures/):
- Pre-built network templates (FeedForward, VAE, etc.)

### Type-Driven Layer Connection

The system uses **type matching** to automatically create connective layers:

1. When `NeuralNetwork` is constructed with a list of layers, it iterates through adjacent pairs
2. Calls `get_connective_layer(input_layer, output_layer)` for incompatible adjacent layers
3. Searches `connective_layers_list` for a connective class where:
   - `ConnectiveClass.input_class` matches the input layer type
   - `ConnectiveClass.output_class` matches the output layer type
4. Falls back to `BlankConnective` (no visual) if no match found

This enables:
- Seamless composition: `NeuralNetwork([ConvLayer(...), FeedForwardLayer(...)])` automatically handles Conv-to-Dense connection
- Easy extensibility: Add new connective layer type, it is auto-discovered
- Example: Feeding an ImageLayer into ConvLayer automatically creates `ImageToConvolutional2DLayer`

### Data Flow in Forward Pass

1. User calls `nn.make_forward_pass_animation()`
2. NeuralNetwork iterates through all layers (input + connective)
3. Calls `layer.make_forward_pass_animation()` on each
4. Collects animations in order
5. Wraps in `Succession` (sequential) or `AnimationGroup` (parallel)
6. Returns composite animation to be played in Manim scene

## Testing Strategy

### Graphical Tests

ManimML uses frame-by-frame visual regression testing via the `@frames_comparison` decorator:

```python
__module_test__ = "feed_forward"  # Required module name

@frames_comparison
def test_FeedForwardScene(scene):
    """Test appearance of feed forward network"""
    nn = NeuralNetwork([FeedForwardLayer(3), FeedForwardLayer(5)])
    scene.add(nn)
```

Control frames are stored as `.npz` files and compared pixel-by-pixel. Use `--set_test` to create baseline frames for new tests.

### Unit Tests

Located in `tests/`:
- Layer instantiation and composition
- Network structure validation
- Activation function behavior
- Decision tree parsing
- Color scheme switching

## Common Development Workflows

### Adding a New Layer Type

1. Create new file in `manim_ml/neural_network/layers/`
2. Inherit from `VGroupNeuralNetworkLayer`
3. Implement `construct_layer()` to build Manim mobjects
4. Implement `make_forward_pass_animation()` to create animation
5. Add to `__init__.py` exports

### Adding a Connective Layer

1. Create file `[InputType]_to_[OutputType].py` in `manim_ml/neural_network/layers/`
2. Inherit from `ConnectiveLayer`
3. Set `input_class` and `output_class` attributes
4. Implement `make_forward_pass_animation()`
5. Import in `manim_ml/neural_network/layers/__init__.py` (auto-discovery)

### Creating Animation Examples

Use the examples in `examples/` as templates. Run with Manim:
```bash
manim -pql examples/readme_example/convolutional_neural_networks.py
```

Flags: `-p` (play), `-q[l|m|h]` (quality: low/medium/high), `-l` (low res)

## Dependencies

- **manim** (Manim Community Library) - Core animation framework
- **numpy** - Array operations
- **PIL/Pillow** - Image handling
- **sklearn** (optional) - Decision tree examples
- **opencv-python** - Image processing (optional)
- **setuptools** - Build system

Install with: `pip install -e .` (installs in development mode with setuptools)

## Notes

- 3D scenes use ThreeDScene and require proper camera configuration via `ManimMLConfig.three_d_config`
- All visualization coordinates are Manim units (typically range -7 to 7)
- Color scheme is global; changing it affects all subsequently created visualizations
- The library is a framework for creating ML visualizations, not for data processing

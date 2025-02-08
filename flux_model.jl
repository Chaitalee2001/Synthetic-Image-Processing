using Flux, Images, CUDA

# Load and preprocess image
function load_image(path)
    img = load(path)  # Load image
    img = imresize(img, (224, 224))  # Resize to match model input
    img = Float32.(channelview(img))  # Normalize and convert to tensor
    return reshape(img, (224, 224, 3, 1))  # Reshape for model (H, W, C, Batch)
end

# Define simple CNN model
model = Chain(
    Conv((3, 3), 3 => 8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(3136, 64, relu),
    Dense(64, 10)  # Output 10 neurons (example, can be any number)
)

# Load preprocessed image and forward pass
img_tensor = load_image("preprocessed_image_1.png")
output = model(img_tensor)

# Print model output
println("Forward pass output: ", output)

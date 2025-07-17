import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import ssl

# This line helps prevent SSL errors when downloading the model for the first time.
ssl._create_default_https_context = ssl._create_unverified_context

class ASCII_Generator_VGG:
    """
    Generates ASCII art by interpreting an image with a pre-trained VGG network.
    """

    def __init__(self):
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        self.feature_map = None
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def capture_feature_map(self, module, input, output):
        self.feature_map = output

    def generate(self, img_path: str, ascii_ramp: str, width: int = 120):
        """
        Generates ASCII art using a specific ASCII character ramp.
        """
        # Set up a hook to capture the feature map from an early layer
        hook = self.model[2].register_forward_hook(self.capture_feature_map)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error opening image {os.path.basename(img_path)}: {e}")
            hook.remove()
            return None

        # Resize image based on desired width and aspect ratio
        original_width, original_height = img.size
        aspect_ratio_correction = 0.55
        new_height = int(width * original_height / original_width * aspect_ratio_correction)
        resized_img = img.resize((width, new_height))

        # Preprocess the image and run it through the model
        input_tensor = self.preprocess(resized_img).unsqueeze(0)
        with torch.no_grad():
            self.model(input_tensor)

        # We have our feature map, so we can remove the hook
        hook.remove()

        if self.feature_map is None:
            print("Error: Feature map was not generated.")
            return None

        # Process the feature map to create the ASCII art
        fm = self.feature_map.squeeze(0)
        intensity_map = torch.mean(fm, dim=0)
        min_val, max_val = torch.min(intensity_map), torch.max(intensity_map)
        normalized_map = (intensity_map - min_val) / (max_val - min_val)

        # Map normalized intensities to characters from the provided ramp
        ascii_str = ""
        for row in normalized_map:
            for value in row:
                char_index = int(value * (len(ascii_ramp) - 1))
                ascii_str += ascii_ramp[char_index]
            ascii_str += "\n"

        return ascii_str.strip()


# --- Main execution block ---
if __name__ == "__main__":
    # Set to True for light backgrounds (e.g., white), False for dark backgrounds
    light_mode = True # <-- Choose here

    # Set the desired width of the ASCII art in characters
    output_width = 120

    # Define the ASCII ramps
    ramp_light_bg = " .:-=+*#%@"  # Chars for dark pixels
    ramp_dark_bg = "@%#*+=:. "    # Chars for light pixels

    # Select the appropriate ramp based on the mode
    selected_ramp = ramp_light_bg if light_mode else ramp_dark_bg

    # Initialize the generator
    print("Initializing VGG model...")
    generator = ASCII_Generator_VGG()

    # Find and process all .png files in the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    files_processed = 0
    print(f"Scanning for *.png files in: {script_directory}\n")

    for filename in os.listdir(script_directory):
        if filename.lower().endswith('.png'):
            files_processed += 1
            print(f"Processing '{filename}'...")

            image_path = os.path.join(script_directory, filename)

            # Generate the art
            ascii_art = generator.generate(image_path, selected_ramp, output_width)

            if ascii_art:
                # Create a corresponding output filename
                output_filename = f"ASCII_{filename.rsplit('.', 1)[0]}.txt"
                output_path = os.path.join(script_directory, output_filename)

                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(ascii_art)
                    print(f"✅ Success! ASCII art saved to '{output_filename}'\n")
                except Exception as e:
                    print(f"❌ Error writing to file: {e}\n")

    if files_processed == 0:
        print("No .png files found in the script's directory. Please add some and run again.")
    else:
        print("All files processed.")

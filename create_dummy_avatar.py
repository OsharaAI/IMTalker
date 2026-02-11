from PIL import Image
import numpy as np

# Create a 256x256 dummy image with random noise (simulating a face for pure pipeline testing)
# In reality, Face Alignment might fail if no face is detected.
# So we should probably try to download a face or hope the user has one?
# Wait, face-alignment will definitely fail if no face.
# I will try to find *any* image in the repo.

def create_dummy():
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save("test_avatar.png")
    print("Created test_avatar.png")

if __name__ == "__main__":
    create_dummy()

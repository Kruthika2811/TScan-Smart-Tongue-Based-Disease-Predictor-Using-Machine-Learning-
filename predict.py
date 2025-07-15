# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the trained model
# # model = load_model('../model/tongue_model.h5')  # Adjust path if needed
# model = load_model('model/tongue_model.h5')


# # List of class names (same order as training)
# class_names = ['anemia', 'gastritis', 'healthy']  # Make sure this matches train_data.class_indices

# # Path to a test image
# img_path = 'test_images/gastritis1.0.jpg'  # Change this to your image path

# # Load and preprocess the image
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# # Predict
# pred = model.predict(img_array)
# predicted_class = class_names[np.argmax(pred)]

# print(f"✅ Predicted Class: {predicted_class}")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model/tongue_model.h5')

# Class names in the same order used during training
class_names = ['anemia', 'gastritis', 'healthy']

# Path to test image
img_path = 'test_images/gastritis1.0.jpg'  # Replace with your actual image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
predicted_index = np.argmax(pred)
predicted_class = class_names[predicted_index]
confidence = round(100 * np.max(pred), 2)

# 🖨️ First print the prediction and explanation
print(f"\n✅ Predicted Class: {predicted_class} ({confidence}%)")

if predicted_class == "anemia":
    print("→ Model detected **anemia** likely due to pale or whitish appearance of the tongue.")
elif predicted_class == "gastritis":
    print("→ Model detected **gastritis** likely due to reddish or patchy areas on the tongue.")
else:
    print("→ Tongue appears **healthy** with normal pink tone and smooth surface.")

# 🖼️ Then show the image after printing output
plt.imshow(img)
plt.axis('off')
plt.title(f"{predicted_class} ({confidence}%)")
plt.show()

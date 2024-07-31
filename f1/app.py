import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
from PIL import Image
from vipas import model
from vipas.exceptions import UnauthorizedException, NotFoundException, RateLimitExceededException
import json

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to 32x32 pixels using LANCZOS resampling
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(image)
    
    # Initialize an 8x8 matrix to hold the counts of "on" pixels
    blocks = img_array.reshape(8, 4, 8, 4).sum(axis=(1, 3))
    
    # Normalize counts to fit in the range 0 to 16
    blocks = (blocks / blocks.max() * 16).astype(int)
    
    return blocks.reshape(1, -1)

# Set the title and description
st.title("🖼️ Handwritten Digit Recognition")
st.markdown("""
    Upload an image and let the model identify it.
    This model can identify the digit .
""")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image_features = preprocess_image(image)

    # Convert numpy int64 to python int
    image_features = image_features.astype(int).tolist()
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger the classification
    if st.button('🔍 Classify Image'):
        vps_model_client = model.ModelClient()
        model_id = "mdl-svxabb9ffpvhx"  # Replace with your model ID

        try:
            api_response = vps_model_client.predict(model_id=model_id, input_data=image_features[0])
            prediction = api_response  # Adjust based on actual response structure
            st.markdown(f"""
                <div style="padding: 1rem; border: 2px solid #4CAF50; border-radius: 10px; background-color: #e7f4e4; color: #2b542c; font-size: 1.5rem; text-align: center; font-weight: bold;">
                    Image classified as: {prediction}
                </div>
            """, unsafe_allow_html=True)
        except UnauthorizedException as e:
            st.error(f"Unauthorized exception: {str(e)}")
        except NotFoundException as e:
            st.error(f"Not found exception: {str(e)}")
        except RateLimitExceededException:
            st.error("Rate limit exceeded exception")
        except Exception as e:
            st.error(f"Exception when calling model->predict: {str(e)}")

# Add some styling with Streamlit's Markdown
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            padding: 0;
        }
        .stApp > header {
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        .stApp > main {
            margin-top: 4rem;
            padding: 2rem;
        }
        .stTitle {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stFileUploader label {
            font-size: 1.1rem;
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .css-1cpxqw2.e1ewe7hr3 {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .stAlert {
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #e7f4e4;
            color: #2b542c;
            font-size: 1.1rem;
            padding: 1rem;
        }
        .stAlert .stMarkdown {
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

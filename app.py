import streamlit as st
import numpy as np

# Set the title of the app
st.title("Random Number Generator using NumPy")

# Add a slider for the user to select the number of random numbers to generate
num = st.slider("Select how many random numbers you want:", min_value=1, max_value=100, value=5)

# Add a button to generate the random numbers
if st.button("Generate Random Numbers"):
    # Generate the random numbers using numpy's np.random
    random_numbers = np.random.randint(0, 100, num)

    # Display the random numbers
    st.write(f"Generated {num} random numbers:")
    st.write(random_numbers)

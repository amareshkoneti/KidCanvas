from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
from gtts import gTTS
import os
import base64
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import io
import stability_sdk.client
from stability_sdk.interfaces.gooseai.generation.generation_pb2 import *

# Initialize Flask app
app = Flask(__name__)


stability_api = stability_sdk.client.StabilityInference(
    key="",  # Replace with your Stability AI API key
    verbose=True,  # Enable logging for debugging
)

# Function to generate and save an image
def generate_image(prompt):
    # Generate an image using the Stability AI API
    answers = stability_api.generate(
        prompt=prompt,
        seed=42,               # Seed for consistency
        steps=50,              # Number of inference steps
        cfg_scale=7.5,         # Prompt adherence
        width=512,             # Image width
        height=512,            # Image height
        samples=1,             # Number of images
        sampler=SAMPLER_K_LMS  # Sampling method
    )

    # Process the response and handle image artifacts
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == ARTIFACT_IMAGE:
                try:
                    # Convert binary data to a PIL image
                    image = Image.open(io.BytesIO(artifact.binary))
                    # Save the image to a file
                    output_path = "static/uploads/generated_image.png"
                    image.save(output_path)
                    return output_path
                except Exception as e:
                    print(f"Error processing image: {e}")
                    return None
    return None


def analyze_image(image_path):
    """
    Analyzes the image and generates a textual description using a pre-trained model.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Use a pre-trained BLIP model for image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process the image and generate a caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


# Set the upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
genai.configure(api_key="")
# Replace with your Vision API key

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to analyze the image using Google Cloud Vision API
def analyze_image(image_path):
    """
    Returns a dictionary with the results.

    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process the image and generate a caption
    inputs = processor(image_path, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    # Encode the image to base64
    image_base64 = encode_image(image_path)

    # Prepare the request payload


  # Parse the response
def analyze_image(image_path):
    """
    Analyzes the image and generates a textual description using a pre-trained model.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Use a pre-trained BLIP model for image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process the image and generate a caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

# Step 2: Story Generation (NLP) 
def generate_story(image_description, age_group="5-8"):
    """
    Generates a kid-friendly story based on the image description and age group.
    """
    # Use a pre-trained GPT-2 model for text generation
    generator = pipeline("text-generation", model="gpt2")

    # Define a more engaging prompt for the story generator
    prompt = (
        f"Once upon a time, there was {image_description}. "
        f"Write a short, magical, and fun story for a {age_group}-year-old child that begins with this description."
    )

    # Generate the story
    story = generator(prompt, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.8)
    return story[0]['generated_text']


# Function to generate a story using Google Gemini API

def classify_emotions(response_text):

    # Example response: "The emotions in the image are happy, excited, and joyful."
    emotions = []
    if "emotions" in response_text.lower():
        # Extract the emotions from the response
        emotion_text = response_text.split("emotions are")[-1].strip().strip(".").strip().replace('*','').strip()
        emotions = [e.strip() for e in emotion_text.split(",")]
    return emotions

def generate_emotions(image_path,language='es'):
    image = Image.open(image_path)
    # Use Gemini to generate the story
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            "Identify the emotions in the given image",
            f"And generate the text in {language}",
            image,
        ]
    )
    emotions = classify_emotions(response.text)
    emotionss = emotions
    return emotionss

def generate_stry(image_path, age_group="5-8",language='en'):

    image = Image.open(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [
            f"Write a short, fun, and imaginative story for a {age_group}-year-old kid based on this image. "
            "Make it engaging, age-appropriate, and full of wonder. "
            "Focus on creativity and a clear beginning, middle, and end. "
            "Avoid using complex words or phrases. "
            "The story should be easy to understand and enjoyable for children.",
            f"And generate the text in {language}",
            image,
        ]
    )
    return response.text
def text_to_speech(text, filename="story.mp3"):
    """
    Converts text to speech and saves it as an audio file.
    """
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

@app.route("/")
def index():
        return render_template("home.html")

# Flask Routes
@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if an image file was uploaded
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]

        # If no file is selected
        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded file with a custom filename
        custom_filename = "image.jpg"  # Fixed filename
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], custom_filename)
        file.save(image_path)
        # Analyze the image
        language = request.form.get("language", "en")
        description = generate_emotions(image_path,language)
        # Generate story

        age_group = request.form.get("age_group", "5-8")
        story = generate_stry(image_path, age_group,language)
        custom_audio = 'stories.mp3'
        audio_filename = text_to_speech(story, os.path.join(app.config["UPLOAD_FOLDER"], custom_audio))

        # Render the result page with the story and analysis results
        return render_template("result.html", story=story, emotions=description, image_filename=custom_filename,audio_filename=custom_audio)

    # Render the upload page
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Use the custom image from the static/generated_images folder
        custom_filename = "generated_image.png"  # The default custom image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], custom_filename)

        # Retrieve form inputs
        language = request.form.get("language", "en")
        age_group = request.form.get("age_group", "5-8")

        # Analyze the custom image
        description = generate_emotions(image_path, language)

        # Generate story
        story = generate_stry(image_path, age_group, language)

        # Generate audio
        custom_audio = 'stories.mp3'
        audio_filename = text_to_speech(story, os.path.join(app.config["UPLOAD_FOLDER"], custom_audio))

        # Render the result page with the story and analysis results
        return render_template("result.html", story=story, emotions=description, image_filename=f"generated_image.png", audio_filename=custom_audio)

    # Render the upload page
    return render_template("index.html")



@app.route("/generate_image", methods=["POST"])
def handle_image_generation():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return "Prompt cannot be empty.", 400

    # Generate the image using the Stability API
    image_path = generate_image(prompt)
    if image_path:
        return render_template("image_result.html", prompt=prompt, image_path=image_path)
    else:
        return "Failed to generate the image. Please try again later.", 500
    

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

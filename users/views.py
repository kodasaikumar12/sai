import json
from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
import os


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from django.shortcuts import render
from django.conf import settings



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import gdown

# Model path
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'resnet34_model.h5')

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.join(settings.BASE_DIR, 'models'), exist_ok=True)

    url = "https://drive.google.com/uc?id=1eVYOlAPnP-s13KzNg2t8jaBPYB9zvsgW"

    print("Downloading AI model from Google Drive...")

    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
# Lazy load model
model = None

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    return model

class_names = ['Intact', 'Damaged']
# Prediction view
def predict_view(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        # Preprocess image
        img = image.load_img(full_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        model = get_model()
        prediction = model.predict(img_array)[0]

        if len(prediction) == 1:  # sigmoid
            predicted_class = class_names[int(prediction > 0.5)]
            confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
        else:  # softmax
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        context = {
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}",
            'image_url': fs.url(file_path),
        }

    return render(request, 'users/predict.html', context)
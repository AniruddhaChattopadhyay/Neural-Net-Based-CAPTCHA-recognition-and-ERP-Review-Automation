from keras.models import load_model
import tensorflow
from helpers import resize_to_fit
import numpy as np
import cv2
import pickle

def break_captcha(img1):
    try:
        from PIL import Image
    except ImportError:
        import Image
    img=cv2.imread(img1,cv2.IMREAD_GRAYSCALE)
    predicted =""
    ret,img = cv2.threshold(img,40,255,cv2.THRESH_BINARY_INV)
    mask= np.zeros(img.shape)
    indexV=45
    if(img[45][4]==255 and img[45][7]==255):
        indexV=47

    indexH=10
    if(img[4][10]==255 and img[7][10]==255):
        indexH=8

    for i in range(img.shape[0]): #removing horizontal lines
        if(img[i][indexH]==255):
            for j in range(img.shape[1]):
                img[i][j]=0
                mask[i][j]=255

    for j in range(img.shape[1]): #removing vertical lines
        if(img[indexV][j]==255):
            for i in range(img.shape[0]):
                img[i][j]=0
                mask[i][j]=255

    img_copy = img.copy()
    kernel = np.ones((3,3),np.uint8)
    #kernel[0][0]=kernel[0][2]=kernel[2][0]=kernel[2][2]=0
    img = cv2.dilate(img,kernel,iterations = 1)

    img_subs =cv2.bitwise_and (mask,mask,mask=img - img_copy)
    img_final = img_copy + img_subs
    ret,img_final = cv2.threshold(img_final,40,255,cv2.THRESH_BINARY_INV)

    # cv2.imshow('Image_Copy',img_copy)
    # cv2.imshow('Image',img)
    # cv2.imshow('Mask',mask)
    # cv2.imshow('dilate',img_subs)
    #cv2.imshow('Image Final',img_final)
    #return pytesseract.image_to_string(img_final)
    MODEL_FILENAME = "captcha_model.hdf5"
    MODEL_LABELS_FILENAME = "model_labels.dat"

    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # Load the trained neural network
    model = load_model(MODEL_FILENAME)
    image = img_final
    # print(image.shape)
    # print(image.shape[0])
    rows = image.shape[0]
    cols = image.shape[1]
    black = []
    for col in range(cols):
        flag_black = 0
        for row in range(rows):
            if (image[row][col] == 0):
                flag_black = 1
                break
            else:
                continue
        if (flag_black == 1):
            black += [1]
        else:
            black += [0]

   # print(black)
    start_black_pixel = []
    stop_black_pixel = []

    for b in range(len(black) - 1):
        if (black[b] == 0) and black[b + 1] == 1:
            start_black_pixel += [b]
        if (black[b] == 1) and black[b + 1] == 0:
            stop_black_pixel += [b + 1]
   # print(start_black_pixel, stop_black_pixel)
    if len(start_black_pixel)!=6 or len(stop_black_pixel)!=6:
        for i in range(len(start_black_pixel)):
            w = stop_black_pixel[i]-start_black_pixel[i]
            wid_f = (float) (w/20)
            wid = (int) (wid_f+.5)
            wide = w/wid
            while wid!=1 :
                iter=0
                start_black_pixel.insert(i+iter+1,start_black_pixel[i]+wide)
                stop_black_pixel.insert(i+iter,start_black_pixel[i]+wide)
                wid=wid-1

    predictions = []
    for i in range(len(start_black_pixel)):
        x = start_black_pixel[i]
        y = 6
        h = 26
        w = stop_black_pixel[i] - start_black_pixel[i]

        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)
        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)
        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        #print(letter)
        predicted+=str(letter)
        predictions.append(letter)
    print(predicted)
    return predicted




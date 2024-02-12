import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
 
img = cv.imread(file_path)
img = cv.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert bgr to grayscale
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
kernel = cv.getStructuringElement(cv.MORPH_RECT,(40,40))
img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
img_norm = img_gray - img_opening
(thresh, img_norm_bw) = cv.threshold(img_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
(thresh, img_without_norm_bw) = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#Kontur Plat
contours_vehicle, hierarchy = cv.findContours(img_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# contours_vehicle, hierarchy = cv.findContours(img_without_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
index_plate_candidate = []
index_counter_contour_vehicle = 0

#motorbike
for contour_vehicle in contours_vehicle:
    x,y,w,h = cv.boundingRect(contour_vehicle)
    aspect_ratio = w/h
    
    if w >= 350 and aspect_ratio <= 6 : 
        index_plate_candidate.append(index_counter_contour_vehicle)

    index_counter_contour_vehicle += 1

img_show_plate = img.copy() 
# img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)
img_show_plate_bw = cv.cvtColor(img_without_norm_bw, cv.COLOR_GRAY2RGB)

if len(index_plate_candidate) == 0:
    print("Plat nomor tidak ditemukan")

elif len(index_plate_candidate) == 1:
    x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
    cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
    
else:
    print('Dapat lokasi plat')

    x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])
    cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
(thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


#Preprosesing Karakter
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (1, 1))


img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel)

contours_plate, hierarchy = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

index_chars_candidate = []
index_counter_contour_plate = 0

img_plate_rgb = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)
img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)
#kontur karakter
if index_chars_candidate == []:
    print('Karakter tidak tersegmentasi')
else:
    score_chars_candidate = np.zeros(len(index_chars_candidate))
    counter_index_chars_candidate = 0
    
    for chars_candidateA in index_chars_candidate:
        xA,yA,wA,hA = cv.boundingRect(contours_plate[chars_candidateA])
        for chars_candidateB in index_chars_candidate:

            if chars_candidateA == chars_candidateB:
                continue
            else:
                xB,yB,wB,hB = cv.boundingRect(contours_plate[chars_candidateB])
                y_difference = abs(yA - yB)

                if y_difference < 11:
                    score_chars_candidate[counter_index_chars_candidate] = score_chars_candidate[counter_index_chars_candidate] + 1 

        counter_index_chars_candidate += 1

    print(score_chars_candidate)

    index_chars = []
    chars_counter = 0

    # for score in score_chars_candidate:
    #     if score == max(score_chars_candidate):
    #         index_chars.append(index_chars_candidate[chars_counter])
    #     chars_counter += 1
    
    for score in score_chars_candidate:
        if score >= 1 or score == max(score_chars_candidate):
            index_chars.append(index_chars_candidate[chars_counter])
        chars_counter += 1

    img_plate_rgb2 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)

    for char in index_chars:
        x, y, w, h = cv.boundingRect(contours_plate[char])
        cv.rectangle(img_plate_rgb2,(x,y),(x+w,y+h),(0,255,0),5)
        cv.putText(img_plate_rgb2, str(index_chars.index(char)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)
    
    x_coors = []

    for char in index_chars:
        x, y, w, h = cv.boundingRect(contours_plate[char])
        x_coors.append(x)

    x_coors = sorted(x_coors)
    index_chars_sorted = []

    for x_coor in x_coors:
        for char in index_chars:
            x, y, w, h = cv.boundingRect(contours_plate[char])
            
            if x_coors[x_coors.index(x_coor)] == x:
                index_chars_sorted.append(char)

    img_plate_rgb3 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)

    for char_sorted in index_chars_sorted:
        x,y,w,h = cv.boundingRect(contours_plate[char_sorted])
        cv.rectangle(img_plate_rgb3,(x,y),(x+w,y+h),(0,255,0),5)
        cv.putText(img_plate_rgb3, str(index_chars_sorted.index(char_sorted)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)

img_height = 40
img_width = 40

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model = keras.models.load_model('cnn_model')
num_plate = []

for char_sorted in index_chars_sorted:
    x, y, w, h = cv.boundingRect(contours_plate[char_sorted])
    char_crop = cv.cvtColor(img_plate_bw[y:y+h, x:x+w], cv.COLOR_GRAY2BGR)
    char_crop = cv.resize(char_crop, (img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(char_crop)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    detected_char = class_names[np.argmax(score)]
    num_plate.append(detected_char)

# Combine all characters into the final plate number
plate_number = ''.join(num_plate)

# Print all detected characters together
print(f'Detected characters: {plate_number}')

# Display the final plate number image
cv.putText(img_show_plate, plate_number, (x_plate, y_plate + h_plate + 50), cv.FONT_ITALIC, 2.0, (0, 255, 0), 3)
plt.imshow(cv.cvtColor(img_show_plate, cv.COLOR_BGR2RGB))
plt.title(plate_number)
plt.show()

cv.waitKey(0)

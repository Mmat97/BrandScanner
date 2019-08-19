# BrandScanner
BrandScanner

## BrandScanner
BrandScanner





## Advantages
1.Simply place text in front of camera without having to click anything 
2. Implements frozen_east+text_detection 









## Tools/Languages/References
**Language:** 

[Python3.6](https://www.python.org/)

**Libraries:** 

[NumPy](https://www.numpy.org/)

[OpenCV](https://opencv.org/)

**OS Used:** 

-Mac OS(Terminal)

**Editors:** 
-Visual Studio Code 
-vim

**References**

[Install OpenCV](https://www.codingforentrepreneurs.com/blog/opencv-python-web-camera-quick-test/)

[Text Detection](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py)




## Installation and Run 
**Install:** 

In terminal,once in directory with files,

1.Follow the instructions in the given reference to install opencv and numpy:
[Install OpenCV](https://www.codingforentrepreneurs.com/blog/opencv-python-web-camera-quick-test/)


2.Install Tesseract OCR(Opticsl Character Recognition)

```
	brew install tesseract --HEAD
```

```
	python3 -m pip install pytesseract
```

```
	python3 -m pip install imutils
```




**Run:** 

To start, in file directory, 

```
	python3 camera-test2.py --east frozen_east_text_detection.pb 
```







## Functions

cap = cv2.VideoCapture(0): turn on camera 

cv2.imshow(kWinName,frame): show current frame 





## Example

BrandScanner : n






## Important Notes:

**NOTE 1:** 





##Common Errors

1.bashprofile do-make sure evrything correct and then virtualenv module appears
2. python camera-test.py return ImportError(No module named cv2) because opencv connected
to python3 only 
3. Text recognition is not accurate and must be improved





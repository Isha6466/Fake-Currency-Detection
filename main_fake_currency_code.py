import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib import pyplot as plt
from tkmacosx import Button


def rs2000():
    def Front():
        imgi=cv.imread("Indian currency dataset v1/test/WhatsApp Image 2022-12-02 at 4.42.30 PM.jpeg",0)
        img2i=cv.imread("Indian currency dataset v1/test/WhatsApp Image 2022-12-02 at 4.42.30 PM.jpeg")
        img = cv.imread("Indian currency dataset v1/test/2000__8.jpg",0)
        img3 = cv.imread("Indian currency dataset v1/test/2000__8.jpg")

        imgi2=cv.resize(img2i,(1291,519))
        imgi3=cv.resize(imgi,(1291,519))
 
        crop_image1= img3[0:1000, 750:850]
        hsv = cv.cvtColor(crop_image1, cv.COLOR_BGR2HSV)
        

        crop_image2=img[0:130,390:510]#circle
        crop_image3=img[80:290,5:60]#last
        crop_image4=img[90:430,810:930]#text
        crop_image5=img[300:430,950:1180]#2000
        crop_image6=img[290:430,1180:1400]#tiger
        crop_image7=img[230:300,1180:1400]#2000inbox
        crop_image8=img[100:270,1250:1400]#lastline
        # plt.imshow( hsv),plt.show()
        # plt.imshow( crop_image2),plt.show()
        # plt.imshow( crop_image3),plt.show()
        # plt.imshow( crop_image4),plt.show()
        # plt.imshow( crop_image5),plt.show()
        # plt.imshow( crop_image6),plt.show()
        # plt.imshow( crop_image7),plt.show()
        # plt.imshow( crop_image8),plt.show()

        hsv2=cv.cvtColor(imgi2, cv.COLOR_BGR2HSV)
  
# flags=cv.IMREAD_COLOR
# kernel = np.array([[0, -1, 0],  [-1,5,-1],   [0, -1, 0]])
# image_sharp = cv.filter2D(src=crop_image7, ddepth=-1, kernel=kernel)
# plt.imshow( image_sharp),plt.show()
 #for i in crop_image:
        image8bit = cv.normalize(hsv2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(hsv, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches1 = []
        for m1, m2 in matches:
         if m1.distance < 0.7 * m2.distance:
           good_matches1.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
        cv.imshow("f",flann_matches)
        print("Length:",len(good_matches1))
 #2
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches2 = []
        for m1, m2 in matches:
         if m1.distance < 0.5 * m2.distance:
          good_matches2.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
        cv.imshow("f1",flann_matches)
        print("Length:",len(good_matches2))
 
 #3
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches3 = []
        for m1, m2 in matches:
         if m1.distance < 0.5 * m2.distance:
          good_matches3.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
        cv.imshow("f3",flann_matches)
        print("Length:",len(good_matches3))
 
 #4
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches4 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
           good_matches4.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
        #cv.imshow("f4",flann_matches) 
        print("Length:",len(good_matches4))
 
 #5
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image5, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches5 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
            good_matches5.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches5, None, flags=2)
        #cv.imshow("f5",flann_matches)
        print("Length:",len(good_matches5))
 
 #6
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image6, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches6 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
            good_matches6.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches6, None, flags=2)
        #cv.imshow("f6",flann_matches)
        print("Length:",len(good_matches6))
 
 #7
        image8bit = cv.normalize(imgi3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image7, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches7 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
           good_matches7.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches7, None, flags=2)
        #cv.imshow("f7",flann_matches)
        print("Length:",len(good_matches7))
 
        if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0 or len(good_matches4)==0 or len(good_matches5)==0  ):
           tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
        else:
          tk.messagebox.showinfo(title=None, message="Currency is REAL!")
    
    def Back():
        imgi=cv.imread("Indian currency dataset v1/test/2000back.jpg",0)
        img2i=cv.imread("Indian currency dataset v1/test/2000back.jpg")
        img = cv.imread("Indian currency dataset v1/test/2000back.jpg",0)
        img3 = cv.imread("Indian currency dataset v1/test/2000back.jpg")

        imgi2=cv.resize(imgi,(1291,519))
        crop_image1=img[200:300, 700:750]
        crop_image2=img[100:280,230:310]#text
        crop_image3=img[100:290,90:230]#chashma
        crop_image4=img[290:330,235:655]#elephantb...
        crop_image5=img[90:230,640:720]#circle
        crop_image6=img[89:260,400:650]
        # plt.imshow( crop_image1),plt.show()
        # plt.imshow( crop_image3),plt.show()
        # plt.imshow( crop_image5),plt.show()
        # plt.imshow( crop_image6),plt.show()
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches1 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
            good_matches1.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
        #cv.imshow("f1",flann_matches)
        print("Length:",len(good_matches1))
 #2
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches2 = []
        for m1, m2 in matches:
          if m1.distance < 0.3 * m2.distance:
           good_matches2.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
        #cv.imshow("f2",flann_matches)
        print("Length:",len(good_matches2))
 
 #3
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches3 = []
        for m1, m2 in matches:
         if m1.distance < 0.5 * m2.distance:
          good_matches3.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
        #cv.imshow("f3",flann_matches)
        print("Length:",len(good_matches3))
 
 #4
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches4 = []
        for m1, m2 in matches:
          if m1.distance < 0.5 * m2.distance:
           good_matches4.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
        #cv.imshow("f4",flann_matches)
        print("Length:",len(good_matches4))
 
 #5
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image5, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches5 = []
        for m1, m2 in matches:
          if m1.distance < 0.7 * m2.distance:
           good_matches5.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches5, None, flags=2)
        #cv.imshow("f5",flann_matches)
        print("Length:",len(good_matches5))
 
 #6
        image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        image8biti = cv.normalize(crop_image6, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        sift = cv.xfeatures2d.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        good_matches6 = []
        for m1, m2 in matches:
         if m1.distance < 0.5 * m2.distance:
           good_matches6.append([m1])
        flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches6, None, flags=2)
        #cv.imshow("f6",flann_matches)
        print("Length:",len(good_matches6))

 
 
        if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0 or len(good_matches4)==0 or len(good_matches5)==0 or len(good_matches6)==0  ):
          tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
        else:
           tk.messagebox.showinfo(title=None, message="Currency is REAL!")
           
    root = tk.Tk()
    Entry = tk.Entry(root)
    Entry.pack()
    label1 = tk.Label(root,text="Which Currency you want to check:")
    label1.pack(side=tk.TOP,expand=True)
    button1 = tk.Button(root, text="2000 Front", fg="green",command=Front)
    button1.pack(side=tk.LEFT,expand=True)
    button2 = tk.Button(root, text="2000 Back", fg="red",command=Back)
    button2.pack(side=tk.RIGHT,expand=True)
    root.mainloop()

def rs500():
    def Front5():
     img = cv.imread("Indian currency dataset v1/test/500.jpg",0)
     imgi=cv.imread("Indian currency dataset v1/test/WhatsApp Image 2022-12-03 at 11.14.23 AM.jpeg",0)
     img2=cv.imread("Indian currency dataset v1/test/500.jpg")
     imgi3=cv.imread("Indian currency dataset v1/test/WhatsApp Image 2022-12-03 at 11.14.23 AM.jpeg")
     imgi2=cv.resize(imgi,(1291,519))
 
     crop_image1= img2[0:850, 950:1000]
     hsv = cv.cvtColor(crop_image1, cv.COLOR_BGR2HSV)

     crop_image2=img[100:170,550:730]#circle
     crop_image3=img[190:380,0:90]#last
     crop_image4=img[150:660,1020:1190]
     crop_image5=img[450:650,1250:1550]
     crop_image6=img[430:700,1580:1700]
     crop_image7=img[450:600,100:200]
    #  crop_image8=img[600:700,105:400]
    #  plt.imshow( hsv),plt.show()
    #  plt.imshow( crop_image2),plt.show()
    #  plt.imshow( crop_image3),plt.show()
    #  plt.imshow( crop_image4),plt.show()
    #  plt.imshow( crop_image5),plt.show()
    #  plt.imshow( crop_image6),plt.show()
    #  plt.imshow( crop_image7),plt.show()
    #  plt.imshow( crop_image8),plt.show()

     hsv2=cv.cvtColor(imgi3, cv.COLOR_BGR2HSV)
  
# flags=cv.IMREAD_COLOR
# kernel = np.array([[0, -1, 0],  [-1,5,-1],   [0, -1, 0]])
# image_sharp = cv.filter2D(src=crop_image7, ddepth=-1, kernel=kernel)
# plt.imshow( image_sharp),plt.show()
 #for i in crop_image:
     image8bit = cv.normalize(hsv2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(hsv, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches1 = []
     for m1, m2 in matches:
      if m1.distance < 0.8 * m2.distance:
        good_matches1.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
     #cv.imshow("f1",flann_matches)
     print("Length:",len(good_matches1))
 #2
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches2 = []
     for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
       good_matches2.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
     #cv.imshow("f2",flann_matches)
     print("Length:",len(good_matches2))
 
 #3
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches3 = []
     for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
       good_matches3.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
     #cv.imshow("f3",flann_matches)
     print("Length:",len(good_matches3))
 
 #4
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches4 = []
     for m1, m2 in matches:
       if m1.distance < 0.5 * m2.distance:
        good_matches4.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
     #cv.imshow("f4",flann_matches)
     print("Length:",len(good_matches4))
 
 #5
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image5, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches5 = []
     for m1, m2 in matches:
      if m1.distance < 0.6 * m2.distance:
       good_matches5.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches5, None, flags=2)
     #cv.imshow("f5",flann_matches)
     print("Length:",len(good_matches5))
 
 #6
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image6, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches6 = []
     for m1, m2 in matches:
      if m1.distance < 0.7 * m2.distance:
       good_matches6.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches6, None, flags=2)
     #cv.imshow("f6",flann_matches)
     print("Length:",len(good_matches6))
 
 #7
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image7, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches7 = []
     for m1, m2 in matches:
       if m1.distance < 0.5 * m2.distance:
        good_matches7.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches7, None, flags=2)
     #cv.imshow("f7",flann_matches)
     print("Length:",len(good_matches7))
 
 #8
    #  image8bit = cv.normalize(crop_image8, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    #  image8biti = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    #  sift = cv.xfeatures2d.SIFT_create()
    #  keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    #  keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    #  FLAN_INDEX_KDTREE = 0
    #  index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    #  search_params = dict (checks=50)
    #  flann = cv.FlannBasedMatcher(index_params, search_params)
    #  matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    #  good_matches8 = []
    #  for m1, m2 in matches:
    #    if m1.distance < 0.7 * m2.distance:
    #     good_matches8.append([m1])
    #  flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches8, None, flags=2)
    #  #plt.imshow(flann_matches),plt.show()
    #  print("Length:",len(good_matches8))
     if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0 or len(good_matches4)==0 or len(good_matches5)==0 or len(good_matches6)==0 or len(good_matches7)==0  ):
       tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
     else:
       tk.messagebox.showinfo(title=None, message="Currency is REAL!")

    def Back5():
     img = cv.imread("Indian currency dataset v1/test/500__37.jpg",0)
     imgi=cv.imread("Indian currency dataset v1/test/500__37.jpg",0)
     img2=cv.imread("Indian currency dataset v1/test/500.jpg")
     imgi3=cv.imread("Indian currency dataset v1/test/500__37.jpg")
 
     crop_image1=img[100:260,550:630]#strip2000
     crop_image2=img[250:320,600:690]#text
     crop_image3=img[200:260,610:660]#chashma
#  crop_image4=img[290:330,235:655]#elephantb...
#  crop_image5=img[90:230,640:720]#circle
#  crop_image6=img[89:260,400:650]#sat
#  crop_image7=img[230:300,1180:1400]#2000inbox
#  crop_image8=img[100:270,1250:1400]#lastline
    #  plt.imshow( crop_image1),plt.show()
    #  plt.imshow( crop_image2),plt.show()
    #  plt.imshow( crop_image3),plt.show()
#  plt.imshow( crop_image4),plt.show()
#  plt.imshow( crop_image5),plt.show()
#  plt.imshow( crop_image6),plt.show()

     #hsv2=cv.cvtColor(imgi2, cv.COLOR_BGR2HSV)
  
# flags=cv.IMREAD_COLOR
# kernel = np.array([[0, -1, 0],  [-1,5,-1],   [0, -1, 0]])
# image_sharp = cv.filter2D(src=crop_image7, ddepth=-1, kernel=kernel)
# plt.imshow( image_sharp),plt.show()
 #for i in crop_image:
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches1 = []
     for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches1.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
     #cv.imshow("f1",flann_matches)
     print("Length:",len(good_matches1))
#  #2
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches2 = []
     for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches2.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
     #cv.imshow("f2",flann_matches)
     print("Length:",len(good_matches2))
 #3
     image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
     sift = cv.xfeatures2d.SIFT_create()
     keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
     keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
     FLAN_INDEX_KDTREE = 0
     index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
     search_params = dict (checks=50)
     flann = cv.FlannBasedMatcher(index_params, search_params)
     matches = flann.knnMatch (descriptors1, descriptors2,k=2)
     good_matches3 = []
     for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches3.append([m1])
     flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
     #cv.imshow("f3",flann_matches)
     print("Length:",len(good_matches3))
 
 #4
#  image8bit = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  image8biti = cv.normalize(imgi2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  sift = cv.xfeatures2d.SIFT_create()
#  keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
#  keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
#  FLAN_INDEX_KDTREE = 0
#  index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
#  search_params = dict (checks=50)
#  flann = cv.FlannBasedMatcher(index_params, search_params)
#  matches = flann.knnMatch (descriptors1, descriptors2,k=2)
#  good_matches4 = []
#  for m1, m2 in matches:
#    if m1.distance < 0.5 * m2.distance:
#     good_matches4.append([m1])
#  flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
#  plt.imshow(flann_matches),plt.show()
#  print("Length:",len(good_matches4))
 
#  #5
#  image8bit = cv.normalize(crop_image5, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  image8biti = cv.normalize(imgi2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  sift = cv.xfeatures2d.SIFT_create()
#  keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
#  keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
#  FLAN_INDEX_KDTREE = 0
#  index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
#  search_params = dict (checks=50)
#  flann = cv.FlannBasedMatcher(index_params, search_params)
#  matches = flann.knnMatch (descriptors1, descriptors2,k=2)
#  good_matches5 = []
#  for m1, m2 in matches:
#    if m1.distance < 0.5 * m2.distance:
#     good_matches5.append([m1])
#  flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches5, None, flags=2)
#  plt.imshow(flann_matches),plt.show()
#  print("Length:",len(good_matches5))
 
#  #6
#  image8bit = cv.normalize(crop_image6, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  image8biti = cv.normalize(imgi2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
#  sift = cv.xfeatures2d.SIFT_create()
#  keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
#  keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
#  FLAN_INDEX_KDTREE = 0
#  index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
#  search_params = dict (checks=50)
#  flann = cv.FlannBasedMatcher(index_params, search_params)
#  matches = flann.knnMatch (descriptors1, descriptors2,k=2)
#  good_matches6 = []
#  for m1, m2 in matches:
#    if m1.distance < 0.5 * m2.distance:
#     good_matches6.append([m1])
#  flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches6, None, flags=2)
#  plt.imshow(flann_matches),plt.show()
#  print("Length:",len(good_matches6))

 
 
     if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0   ):
        tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
     else:
       tk.messagebox.showinfo(title=None, message="Currency is REAL!")
    
    root = tk.Tk()
    Entry = tk.Entry(root)
    Entry.pack()
    label1 = tk.Label(root,text="Which Currency you want to check:")
    label1.pack(side=tk.TOP,expand=True)
    button1 = tk.Button(root, text="500 Front", fg="green",command=Front5)
    button1.pack(side=tk.LEFT,expand=True)
    button2 = tk.Button(root, text="500 Back", fg="red",command=Back5)
    button2.pack(side=tk.RIGHT,expand=True)
    root.mainloop()

def rs200():
  def Front200():
    img = cv.imread("Indian currency dataset v1/test/200.jpg",0)
    img2 = cv.imread("Indian currency dataset v1/test/200.jpg")
    
    imgi=cv.imread("Indian currency dataset v1/test/200.jpg",0)
    imgi3=cv.imread("Indian currency dataset v1/test/200.jpg")

    imgi2=cv.resize(imgi,(1291,519))
 
    crop_image1= img2[50:500, 650:750]
    hsv = cv.cvtColor(crop_image1, cv.COLOR_BGR2HSV)
    crop_image2=img[30:180,400:610]#circle
    crop_image3=img[80:290,5:60]#last
    crop_image4=img[259:500,1170:1290]
    crop_image5=img[110:498,750:880]
    # crop_image6=img[430:700,1580:1700]
    # crop_image7=img[340:500,930:1150]
    # crop_image8=img[600:700,105:400]
    # plt.imshow( hsv),plt.show()
    # plt.imshow( crop_image2),plt.show()
    # plt.imshow( crop_image3),plt.show()
    # plt.imshow( crop_image4),plt.show()
    # plt.imshow( crop_image5),plt.show()
    # plt.imshow( crop_image6),plt.show()
    # plt.imshow( crop_image7),plt.show()
    # plt.imshow( crop_image8),plt.show()
    hsv2=cv.cvtColor(imgi3, cv.COLOR_BGR2HSV)
    
    image8bit = cv.normalize(hsv2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(hsv, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches1 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches1.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
    #cv.imshow("f",flann_matches)
    print("Length:",len(good_matches1))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches2 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches2.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches2))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches3 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches3.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
    #cv.imshow("f3",flann_matches)
    print("Length:",len(good_matches3))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches4 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches4.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches4))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image5, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches5 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches5.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches5, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches5))
    
    if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0   ):
        tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
    else:
       tk.messagebox.showinfo(title=None, message="Currency is REAL!")
    
  def Back200():
    
    imgs = cv.imread("Indian currency dataset v1/test/200backi.webp")
    img = cv.imread("Indian currency dataset v1/test/200backi.webp",0)

    imgi=cv.imread("Indian currency dataset v1/test/200backi.webp",0)
#imgc = cv.imread("Indian currency dataset v1/test/2000__8.jpg",0)
    imgii=cv.imread("Indian currency dataset v1/test/200backi.webp",0)

    # imgi2=cv.resize(imgi,(1291,519))
    # imgii2=cv.resize(imgii,(1291,519))
    
    crop_image1=imgs[300:430,1300:1370]
    crop_image2=imgs[190:330,1100:1370]
    crop_image3=imgs[230:550,90:230]
    crop_image4=imgs[150:590,500:650]
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches1 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches1.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches1, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches1))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches2 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches2.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches2, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches2))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image3, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches3 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches3.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches3, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches3))
    
    image8bit = cv.normalize(imgi, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8biti = cv.normalize(crop_image4, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image8bit, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image8biti, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2,k=2)
    good_matches4 = []
    for m1, m2 in matches:
      if m1.distance < 0.5 * m2.distance:
        good_matches4.append([m1])
    flann_matches =cv.drawMatchesKnn(image8bit, keypoints1, image8biti, keypoints2, good_matches4, None, flags=2)
    #plt.imshow(flann_matches),plt.show()
    print("Length:",len(good_matches4))
    if(len(good_matches1)==0 or len(good_matches2)==0 or len(good_matches3)==0   ):
        tk.messagebox.showinfo(title=None, message="Currency is FAKE!")
    else:
       tk.messagebox.showinfo(title=None, message="Currency is REAL!")
  root = tk.Tk()
  Entry = tk.Entry(root)
  Entry.pack()
  label1 = tk.Label(root,text="Which Currency you want to check:")
  label1.pack(side=tk.TOP,expand=True)
  button1 = tk.Button(root, text="200 Front", fg="green",command=Front200)
  button1.pack(side=tk.LEFT,expand=True)
  button2 = tk.Button(root, text="100 Back", fg="red",command=Back100)
  button2.pack(side=tk.RIGHT,expand=True)
  root.mainloop()       
    
    
root = tk.Tk()
Entry = tk.Entry(root)
Entry.pack()
root.geometry('200x150')
label1 = tk.Label(root,text="FAKE CURRENCY DETECTION",font =("Arial Bold",35))
label1.place(x=100,y=10)
label1.pack(side=tk.TOP,expand=True)
label1 = tk.Label(root,text="Which Currency you want to detect:",font =("Arial Bold",25))
label1.place(x=50,y=20)
label1.pack(side=tk.TOP,expand=True)
button1 = tk.Button(root, text="100 ",bg='#ADEFD1',fg='#00203F',command=rs100,highlightbackground='#3E4149')
button1.pack(side=tk.LEFT,padx=10, pady=(15,0),expand=True)
button2 = tk.Button(root, text="200 ", fg="red",command=rs200)
button2.pack(side=tk.RIGHT,padx=20, pady=(15,0),expand=True)
button3 = tk.Button(root, text="500 ", fg="blue",command=rs500)
button3.pack(side=tk.RIGHT,padx=30, pady=(15,0),expand=True)
button4 = tk.Button(root, text="2000 ", fg="purple",command=rs2000)
button4.pack(side=tk.RIGHT,padx=40, pady=(15,0),expand=True)
root.mainloop()


cv.waitKey()
cv.destroyAllWindows()
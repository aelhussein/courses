# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:58:01 2021

@author: Ahmed Elhussein
"""

pupils = np.linspace(1,90,90)
success =0
for i in range(1000):    
    np.random.shuffle(pupils)
    class_1 = pupils[:30]
    class_2 = pupils[30:60]
    class_3 = pupils[60:]
    if 86 in class_1 and 8 in class_1:
        success +=1
    elif 86 in class_2 and 8 in class_2:
        success+=1
    elif 86 in class_3 and 8 in class_3:
        success+=1


success/1000

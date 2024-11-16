#文件的地址
file_path": "runs/local_img/left/big",
    "runs/local_img/left/big"为图片存储的地址
#筛选图片
"select": 1,
    "select"值为1的时候开启筛选
"start": 1,
    从第"start"张开始
"end": 1,
    到第"end"张结束
#在定位的方法中
def location_hfy(self,src):
 "location_hfy":{ 
##在霍夫圆检测的代码中
circles1 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, mD, param2=p2, minRadius=miR, maxRadius=maR)
    "HoughCircles": {
            "minDist": 1000,
            md="minDist",表示两个圆的最小距离
            "param2": 30,
            p2="param2",表示圆的筛选阈值
            "minRadius": 80,
            miR="minRadius",表示圆的最小半径
            "maxRadius": 160
            maR="maxRadius",表示圆的最大半径
    }
}
#矫正的方法中
def correct(self,src,C):
"correct": {
        "ang": -3
        ang,的值表示压力表像逆时针方向旋转的度数
    },
#直线拟合的方法中
def fit_line(self,src, C):
"fit_line": {
##在第一个阈值处理的代码中
ret, img1 = cv2.threshold(img1, thr, 255, cv2.THRESH_BINARY)
    "threshold1": {
        "thresh": 80
        thr="thresh",表示该代码中的阈值
    },
##在第一个拟合指针直线霍夫直线的代码中
lines = cv2.HoughLines(end1, 1, np.pi / 180, th)
    "HoughLines1": {
        "threshold": 20
        th="threshold",表示第一个霍夫直线的阈值
    },
##在第一个筛选指针直线的判断代码中
if (db1-db2) > (rho**2)*fe or (db1-db2) < (rho**2)/fe:
    "f1_error": 1.01,
    fe="f1_error",拟合指针直线表示可容纳的误差
##在第二个阈值处理的代码中
ret, img2 = cv2.threshold(img2, thr, 255, cv2.THRESH_BINARY)
    "threshold2": {
        "thresh": 80
    thr="thresh",表示该代码中的阈值
    },
##在第二个拟合尾针直线霍夫直线的代码中
lines = cv2.HoughLines(end2, 1, np.pi / 180, th)
    "HoughLines2": {
        "threshold": 10
        th="threshold",表示第二个霍夫直线的阈值
    },
##在第二个筛选尾针直线的判断代码中
if abs(the - math.degrees(lineN[0][1])) > (90-fe) and abs(the - math.degrees(lineN[0][1])) < (90+fe)
    "f2_error": 0.5
    fe="f1_error",拟合尾针直线表示可容纳的误差
}
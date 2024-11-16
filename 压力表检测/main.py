import json
import os
import cv2
import math
import numpy as np
from multiprocessing import Manager

class Manometer_1(object):
    def cv_imread(self, file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        # im decode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        return cv_img

    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)
        img_Path = dit["file_path"]
        img_file = os.listdir(img_Path)
        i = 0
        for Path in img_file:
            i += 1
            path = os.path.join(img_Path, Path)
            if i < dit["start"] and dit["select"] == 1:
                continue
            elif i > dit["end"] and dit["select"] == 1:
                break
            try:
                # 第一步：读取图片
                self.original = self.cv_imread(path)
                print("处理的图片：", path)
                # self.Imgshow(self.original) # 展示原图
                img = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
                # self.Imgshow(img) # 展示灰度图

                # 第二步：用霍夫圆定位表的位置和表的中心
                img1, cir = self.location_hfy(img)
                # self.Imgshow(img1)

                # 第三步：矫正
                img2 = self.correct(img1, cir)

                # 第四步：直线拟合
                lineN = self.fit_line(img2, cir)

                # 第五步：刻度读取
                s = self.read_scale(lineN[0], lineN[1], cir)
                print("压力表度数:", round(s, 4), "度")

                # 展示处理的图像
                self.Imgshow(self.original)
            except:
                i += 1
                print("图片处理错误")
                continue

    def location_hfy(self, src):
        img = cv2.Canny(src, 50, 150, apertureSize=3)
        mD = dit["location_hfy"]["HoughCircles"]["minDist"]
        p2 = dit["location_hfy"]["HoughCircles"]["param2"]
        miR = dit["location_hfy"]["HoughCircles"]["minRadius"]
        maR = dit["location_hfy"]["HoughCircles"]["maxRadius"]
        circles1 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, mD, param2=p2, minRadius=miR, maxRadius=maR)
        circles1 = np.uint16(np.around(circles1))
        circle1 = circles1[0][0]
        r0_2 = int(circle1[2] * 0.2)
        black1 = np.zeros((src.shape[0], src.shape[1]), np.uint8)
        cv2.circle(black1, (circle1[0], circle1[1]), r0_2, (255, 255, 255), -1)
        img1 = cv2.bitwise_and(src, black1)
        img1 = img1 + 255
        yl = circle1[1] + r0_2
        yh = circle1[1] - r0_2
        xl = circle1[0] + r0_2
        xh = circle1[0] - r0_2
        for y in range(yh, yl):
            for x in range(xh, xl):
                if img1[y][x] <= 100 and img1[y][x] > 50:
                    img1[y][x] = img1[y][x] * 2
                elif img1[y][x] <= 50:
                    img1[y][x] = 0
        roi = img1[circle1[1] - r0_2:circle1[1] + r0_2, circle1[0] - r0_2:circle1[0] + r0_2]
        roi = cv2.resize(roi, None, fx=5, fy=5)
        retval, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
        roi = cv2.erode(roi, self.kernel, iterations=3)
        x = roi.shape[0]
        y = roi.shape[1]
        for i in range(x):
            for j in range(y):
                if roi[i][j] == 0:
                    break
                else:
                    roi[i][j] = 0
            for j in range(y):
                if roi[i][y - j - 1] == 0:
                    break
                else:
                    roi[i][y - j - 1] = 0
        img = cv2.Canny(roi, 50, 150, apertureSize=3)
        circles2 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, mD, param2=15, minRadius=int(r0_2 / 8) * 5,
                                    maxRadius=int(r0_2 / 2) * 5)
        circles2 = np.uint16(np.around(circles2))
        circle2 = circles2[0][0]
        circle2[0] = int(circle2[0] / 5 + circle1[0] - r0_2)
        circle2[1] = int(circle2[1] / 5 + circle1[1] - r0_2)
        circle2[2] = circle1[2]
        cv2.circle(self.original, (circle2[0], circle2[1]), 2, (255, 255, 255), -1)
        black2 = np.zeros((src.shape[0], src.shape[1]), np.uint8)
        cv2.circle(black2, (circle2[0], circle2[1]), circle1[2], (255, 255, 255), -1)
        img2 = cv2.bitwise_and(src, black2)
        cv2.circle(self.original, (circle1[0], circle1[1]), circle1[2], (150, 250, 200), 2)
        return img2, circle2

    def correct(self, src, C):
        ang = dit["correct"]["ang"]
        m = cv2.getRotationMatrix2D((C[0], C[1]), ang, 1)
        dsize = (src.shape[1], src.shape[0])
        img = cv2.warpAffine(src, m, dsize)
        self.original = cv2.warpAffine(self.original, m, dsize)
        return img

    def fit_line(self, src, C):
        black = np.zeros((src.shape[0], src.shape[1]), np.uint8)
        cv2.circle(black, (C[0], C[1]), int(C[2] / 1.5), (255, 255, 255), -1)
        cv2.circle(black, (C[0], C[1]), int(C[2] / 3), (0, 0, 0), -1)
        cv2.rectangle(black, (C[0] - int(C[2] / 8), C[1]), (C[0] + int(C[2] / 8), C[1] + C[2]), (0, 0, 0), -1)
        img1 = cv2.bitwise_and(src, black)
        img1 = img1 + 255
        thr = dit["fit_line"]["threshold1"]["thresh"]
        ret, img1 = cv2.threshold(img1, thr, 255, cv2.THRESH_BINARY)
        img1 = cv2.dilate(img1, self.kernel, iterations=1)
        end1 = cv2.Canny(img1, 50, 150, apertureSize=3)
        th = dit["fit_line"]["HoughLines1"]["threshold"]
        lines = cv2.HoughLines(end1, 1, np.pi / 180, th)
        lineN = []
        db1 = C[0] ** 2 + C[1] ** 2
        find = True
        fe = dit["fit_line"]["f1_error"]
        while (find):
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0, y0 = a * rho, b * rho
                db2 = (x0 - C[0]) ** 2 + (y0 - C[1]) ** 2
                if (db1 - db2) > (rho ** 2) * fe or (db1 - db2) < (rho ** 2) / fe:
                    continue
                lineN.append([rho, theta])
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(self.original, pt1, pt2, (255, 255, 255), 2)
                cv2.line(self.original, (0, 0), (int(x0), int(y0)), (0, 255, 0), 2)
                cv2.line(self.original, (0, 0), (C[0], C[1]), (255, 0, 0), 2)
                cv2.line(self.original, (int(x0), int(y0)), (C[0], C[1]), (0, 0, 255), 2)
                find = False
                break
            fe += 0.01
        black = np.zeros((src.shape[0], src.shape[1]), np.uint8)
        cv2.circle(black, (C[0], C[1]), int(C[2] / 3), (255, 255, 255), int(C[2] / 6), 2)
        img2 = cv2.bitwise_and(src, black)
        img2 = img2 + 255
        thr = dit["fit_line"]["threshold2"]["thresh"]
        ret, img2 = cv2.threshold(img2, thr, 255, cv2.THRESH_BINARY)
        iter = dit["fit_line"]["dilate"]["iterations"]
        img2 = cv2.dilate(img2, self.kernel, iterations=iter)
        end2 = cv2.Canny(img2, 50, 150, apertureSize=3)
        th = dit["fit_line"]["HoughLines2"]["threshold"]
        lines = cv2.HoughLines(end2, 1, np.pi / 180, th)
        find = True
        fe = dit["fit_line"]["f2_error"]
        while (find):
            for line in lines:
                rho, theta = line[0]
                the = math.degrees(theta)
                if abs(the - math.degrees(lineN[0][1])) > (90 - fe) and abs(the - math.degrees(lineN[0][1])) < (
                        90 + fe):
                    lineN.append([rho, theta])
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(self.original, pt1, pt2, (255, 255, 255), 2)
                    cv2.line(self.original, (0, 0), (int(x0), int(y0)), (0, 255, 255), 2)
                    find = False
                    break
            fe += 0.1
        return lineN

    def read_scale(self, l1, l2, C):
        fl = dit["read_scale"]["infinite_limit"]
        if abs(abs(l1[1] - np.pi / 2) - np.pi / 2) <= fl:
            return 0.3
        if abs(abs(l2[1] - np.pi / 2) - np.pi / 2) <= fl:
            return 0.1

        k1 = np.tan(l1[1] - np.pi / 2)
        k2 = np.tan(l2[1] - np.pi / 2)
        x1 = l1[0] * np.cos(l1[1])
        y1 = l1[0] * np.sin(l1[1])
        x2 = l2[0] * np.cos(l2[1])
        y2 = l2[0] * np.sin(l2[1])
        x0 = (y2 - y1 + (k1 * x1) - (k2 * x2)) / (k1 - k2)
        y0 = k1 * (x0 - x1) + y1
        aC = math.atan(C[1] / C[0])
        rC = l1[0] / math.cos(abs(aC - l1[1]))
        xc = rC * np.cos(aC)
        yc = rC * np.sin(aC)
        cv2.circle(self.original, (int(x1), int(y1)), 3, (0, 255, 255), -1)
        cv2.circle(self.original, (int(x2), int(y2)), 3, (0, 255, 255), -1)
        cv2.circle(self.original, (int(x0), int(y0)), 3, (100, 255, 100), -1)
        cv2.circle(self.original, (int(xc), int(yc)), 3, (100, 255, 100), -1)

        if x0 < xc:
            rad = math.degrees(math.atan((xc - x0) / (y0 - yc)))
            if rad < 0:
                rad = 180 + rad
            rad = 135 + rad
        elif x0 == xc:
            rad = 135
        else:
            rad = math.degrees(math.atan((x0 - xc) / (y0 - yc)))
            if rad < 0:
                rad = 180 + rad
            rad = 135 - rad
        kedu = rad * (0.6 / 270)
        return kedu

    def Imgshow(self, img):
        cv2.imshow('result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == "__main__":
    manager = Manager()

    path = r"set/"
    filename = r"runini"
    filesuffix = r".json"

    filePath = path + filename + filesuffix

    with open(filePath, encoding="utf-8") as ru_file:
        dit = manager.dict(**json.load(ru_file))
        m = Manometer_1()


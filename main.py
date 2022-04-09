from gamewindow import Gamewindow
import win32key
import cv2
import logging
import time
import subprocess
import io
import os
import numpy as np
from multiprocessing import Process
from PIL import Image
from matplotlib import pyplot as plt
import random
from mirrormodel import PredictModel

# mrr = Gamewindow("Mirror2  ")
hwnd = Gamewindow("Mirror2  ")
hwnd.ResetWindow()
logging.info('Windows Ready')
mirror_model = PredictModel.from_keras_model('mirror2_model')
# path
work_path = os.path.dirname(__file__)
save_path = os.path.join(work_path, 'captures')
mirror_save_path = os.path.join(save_path, 'mirror2')

# modelfolder_path = os.path.join(work_path, 'mdmodel')
# label_path = os.path.join(modelfolder_path , 'labels.txt')
# model_path = os.path.join(modelfolder_path, 'model.tflite')
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
# print(im.shape)
# cv2.imshow("image", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# hwnd.MouseLefttClickPos((0,0))
def preview(hwnd):
    loop_time = time.time()
    while(True):

        # get an updated image of the game
        im = hwnd.CaptureCV(method='mss', wtype='universal')
        # im = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_RGB2BGR)
        plt.subplot(111), plt.imshow(cv2.cvtColor(
            im ,cv2.COLOR_BGR2RGB)), plt.title('line2')
        plt.xticks([]), plt.yticks([])
        plt.show()
        quit()

        # debug the loop rate
        # print('FPS {}'.format(1 / (time.time() - loop_time)))
        # print(im.shape)
        loop_time = time.time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        # if cv2.waitKey(1) == ord('q'):
        #     cv2.destroyAllWindows()
        #     break


def capture_save(hwnd):
    # mirror_save_path = os.path.join(save_path, 'mirror2')
    os.makedirs(mirror_save_path, exist_ok=True)
    for i in range(60):
        im = hwnd.CaptureCV(method='mss', wtype='universal')
        cv2.imshow('image', im)
        tmp = os.path.join(mirror_save_path, str(i)+'.jpg')
        cv2.imwrite(tmp, im)

    cv2.destroyAllWindows()

def preprocess_shit():
    lefttop, righttop, leftbot, rightbot = [195, 192], [644, 192], [195, 584], [644, 584]

    # for p in points:
    #     if p[0] < tmp_center[0]:
    #         if p[1] < tmp_center[1]:
    #             lefttop = p
    #         else:
    #             leftbot = p
    #     else:
    #         if p[1] < tmp_center[1]:
    #             righttop = p
    #         else:
    #             rightbot = p
    # print(lefttop, righttop, leftbot, rightbot)

    # lefttop[0] = leftbot[0] = int(round(leftbot[0]*1.258))
    # lefttop[1] = righttop[1] = int(round(righttop[1] * 1.4516))
    # leftbot[1] = rightbot[1] = int(round(rightbot[1] * 0.9438))
    # righttop[0] = rightbot[0] = int(round(rightbot[0] * 0.9538))

    # draw new points
    # for p in (lefttop, righttop, leftbot, rightbot):
    #     cv2.circle(line_img2, p, 4, (255, 0, 0), -1)

    # print('data')
    # print(lefttop, righttop, leftbot, rightbot)
    grids = []
    gcol = 8
    grow = 7
    gwidth = int(round(abs(lefttop[0] - righttop[0])/gcol))
    gheight = int(round(abs(lefttop[1] - leftbot[1])/grow))
    # print(gwidth,gheight)

    for row in range(7):
        tmp = []
        for col in range(8):
            tmp.append((lefttop[0] + gwidth * col, lefttop[1] + gheight * row))
            
            # cv2.circle(line_img2, (lefttop[0] + gwidth * col, lefttop[1] + gheight * row) , 3, (0, 0, 255), -1)
            
        grids.append(tmp)
            # result
    # plt.subplot(111), plt.imshow(cv2.cvtColor(
    #     line_img2, cv2.COLOR_BGR2RGB)), plt.title('line2')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return int(gwidth*0.984), gheight, grids
    

def preprocess_img(im):
    

    # Low-level CV techniques (grayscale & blur)
    

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny algorithm
    edges = cv2.Canny(gray_blur, 100, 200)

    # Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    v_lines = []
    h_lines = []
    # line_img = im.copy()
    for line in lines:
        rho, theta = line[0]
        if(np.absolute(np.round(theta, 4)-np.round(np.pi/2, 4)) < 0.0001):
            h_lines.append(line[0])
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            # cv2.line(line_img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif(np.absolute(np.round(theta, 4)-np.round(0, 4)) < 0.0001):
            v_lines.append(line[0])
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a*rho
            # y0 = b*rho
            # x1 = int(x0 + 2000*(-b))
            # y1 = int(y0 + 2000*(a))
            # x2 = int(x0 - 2000*(-b))
            # y2 = int(y0 - 2000*(a))
            # cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cluster lines
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    data = np.array(v_lines).reshape((-1, 2))
    data = np.float32(data)
    _, _, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
    v_lines_clusters = centers.tolist()

    data = np.array(h_lines).reshape((-1, 2))
    data = np.float32(data)
    _, _, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
    h_lines_clusters = centers.tolist()
    
    # draw lines
    # line_img2 = im.copy()
    # for line in (v_lines_clusters + h_lines_clusters):
    #     try:
    #         print(line)
            
    #         rho, theta = line
    #         print(rho, theta)
    #         quit()
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 2000*(-b))
    #         y1 = int(y0 + 2000*(a))
    #         x2 = int(x0 - 2000*(-b))
    #         y2 = int(y0 - 2000*(a))
    #     except Exception as e:
    #         print(v_lines_clusters ,h_lines_clusters)
    #         print(line)
    #         raise e
        # cv2.line(line_img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # draw points
    # for r_h, t_h in h_lines:
    #     for r_v, t_v in v_lines:
    #         a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
    #         b = np.array([r_h, r_v])
    #         inter_point = np.linalg.solve(a, b).astype(int)
    #         cv2.circle(line_img2,inter_point,10, (0,255,0), -1)

    # draw points
    points = []
    for r_h, t_h in h_lines_clusters:
        for r_v, t_v in v_lines_clusters:
            a = np.array([[np.cos(t_h), np.sin(t_h)],
                         [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b).astype(int)
            points.append(inter_point)
            # cv2.circle(line_img2, inter_point, 5, (0, 255, 0), -1)

    # process points
    tmp_center = np.mean(points, axis=0)

    lefttop, righttop, leftbot, rightbot = [0, 0], [0, 0], [0, 0], [0, 0]

    for p in points:
        if p[0] < tmp_center[0]:
            if p[1] < tmp_center[1]:
                lefttop = p
            else:
                leftbot = p
        else:
            if p[1] < tmp_center[1]:
                righttop = p
            else:
                rightbot = p
    print(lefttop, righttop, leftbot, rightbot)

    lefttop[0] = leftbot[0] = int(round(leftbot[0]*1.258))
    lefttop[1] = righttop[1] = int(round(righttop[1] * 1.4516))
    leftbot[1] = rightbot[1] = int(round(rightbot[1] * 0.9438))
    righttop[0] = rightbot[0] = int(round(rightbot[0] * 0.9538))

    # draw new points
    # for p in (lefttop, righttop, leftbot, rightbot):
    #     cv2.circle(line_img2, p, 4, (255, 0, 0), -1)

    print('data')
    print(lefttop, righttop, leftbot, rightbot)
    grids = []
    gcol = 8
    grow = 7
    gwidth = int(round(abs(lefttop[0] - righttop[0])/gcol))
    gheight = int(round(abs(lefttop[1] - leftbot[1])/grow))
    # print(gwidth,gheight)

    for row in range(7):
        tmp = []
        for col in range(8):
            tmp.append((lefttop[0] + gwidth * col, lefttop[1] + gheight * row))
            
            # cv2.circle(line_img2, (lefttop[0] + gwidth * col, lefttop[1] + gheight * row) , 3, (0, 0, 255), -1)
            
        grids.append(tmp)
            # result
    # plt.subplot(111), plt.imshow(cv2.cvtColor(
    #     line_img2, cv2.COLOR_BGR2RGB)), plt.title('line2')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return int(gwidth*0.984), gheight, grids


def generate_board(width,height,grids,im):
    # im_path = os.path.join(mirror_save_path, '40.jpg')
    board = []
    for item in np.array(grids).reshape((-1,2)):
        try:
            crop_img = im[item[1]:item[1]+height, item[0]:item[0]+width]

            # demo_save_path = os.path.join(mirror_save_path, 'demo','log')
            # os.makedirs(demo_save_path , exist_ok=True)
            # t = time.localtime()
            # timestamp = time.strftime('%b_%d_%Y_%H%M%S', t)
            # filename = ('demo'+"_" + timestamp+'_'+str(random.randint(0,9))+'.jpg')
            # filename =  os.path.join(demo_save_path, filename)
            # cv2.imwrite(filename, crop_img)
            
            result , acc = mirror_model.predict(crop_img)
            # if acc < 90.0:
            #     demo_save_path = os.path.join(mirror_save_path, 'demo', 'sample')
            #     os.makedirs(demo_save_path , exist_ok=True)
            #     t = time.localtime()
            #     timestamp = time.strftime('%b_%d_%Y_%H%M%S', t)
            #     filename = ('demo'+"_" + timestamp+'_'+str(random.randint(0,99999))+'.jpg')
            #     filename =  os.path.join(demo_save_path, filename)
            #     cv2.imwrite(filename, crop_img)
            board.append(result)
        except Exception as e:
            error_save_path = os.path.join(mirror_save_path, 'error')
            os.makedirs(error_save_path , exist_ok=True)
            t = time.localtime()
            timestamp = time.strftime('%b_%d_%Y_%H%M%S', t)
            imgname = ('error'+"_" + timestamp+'_'+str(random.randint(0,9999999))+'.jpg')
            cropname = ('crop_error'+"_" + timestamp+'_'+str(random.randint(0,999999))+'.jpg')
            cv2.imwrite(os.path.join(error_save_path, imgname), im)
            print('error')
            print(item[1],item[1]+height, item[0],item[0]+width)
            cv2.imwrite(os.path.join(error_save_path, cropname), crop_img)
            raise e
            
    return np.array(board).reshape((7,8))



def solve_board_optimal(board):
    best_score = 0
    best_gtype = "riya"
    best_g = (0,0)
    best_result = (0,0)
    #find five combo
    for i in range(6, -1 , -1):
        for j in range(7, -1, -1):
            
            if board[i][j] == 'riya' :
                continue
            
                     
            if j > 0 and board[i][j - 1] != 'riya':
                #new board
                new_board = np.copy(board)
                #move
                tmp = new_board[i][j]
                new_board[i][j] = new_board[i][j - 1]
                new_board[i][j - 1] = tmp
                
                #solve
                score , result = solve_optimal(new_board, i, j - 1)
                if score > best_score:
                    best_score = score
                    best_gtype = board[i][j]
                    best_g = (i,j)
                    best_result = result
                
            #top
            if i > 0 and board[i - 1][j] != 'riya':
                #new board
                new_board = np.copy(board)
                #move
                tmp = new_board[i][j]
                new_board[i][j] = new_board[i - 1][j]
                new_board[i - 1][j] = tmp
                
                #solve
                score ,result = solve_optimal(new_board, i - 1, j)
                if score > best_score:
                    best_score = score
                    best_gtype = board[i][j]
                    best_g = (i,j)
                    best_result = result
            #right
            if j < 7 and board[i][j + 1] != 'riya':
                #new board
                new_board = np.copy(board)
                #move
                tmp = new_board[i][j]
                new_board[i][j] = new_board[i][j + 1]
                new_board[i][j + 1] = tmp
                
                #solve
                score ,result = solve_optimal(new_board, i, j + 1)
                if score > best_score:
                    best_score = score
                    best_gtype = board[i][j]
                    best_g = (i,j)
                    best_result = result
            #bottom
            if i < 6 and board[i + 1][j ] != 'riya':
                #new board
                new_board = np.copy(board)
                #move
                tmp = new_board[i][j]
                new_board[i][j] = new_board[i + 1][j]
                new_board[i + 1][j] = tmp
                
                #solve
                result = solve_optimal(new_board, i + 1, j)
                if score > best_score:
                    best_score = score
                    best_gtype = board[i][j]
                    best_g = (i,j)
                    best_result = result
                    
            if board[i][j] == 'purple':
                score = 3
                if score > best_score:
                    best_score = score
                    best_gtype = board[i][j]
                    best_g = (i,j)
                    best_result = (0,0)
                    
                    
    if best_score > 0:
        return best_gtype ,[best_g,best_result]
    return None, None


def solve_optimal(board, i, j):
    score = 0
    #horizontal
    #center
    if j > 0 and j < 7: # find 3
        
        if board[i][j] == board[i][j - 1] and board[i][j] == board[i][j + 1]:
            score += 1
    #left
    if j > 1: # find 3
        if board[i][j] == board[i][j - 1] and board[i][j] == board[i][j - 2]:
            score += 1
    #right
    if j < 6: # find 3
        if board[i][j] == board[i][j + 1] and board[i][j] == board[i][j + 2]:
            score += 1
        
    #vertical
    #center
    if i > 0 and i < 6: # find 3
        if board[i][j] == board[i - 1][j] and board[i][j] == board[i + 1][j] :
            score += 1
    #top
    if i > 1: # find 3
        if board[i][j] == board[i - 1][j] and board[i][j] == board[i - 2][j] :
            score += 1
    #bottom
    if i < 5: # find 3
        if board[i][j] == board[i + 1][j] and board[i][j] == board[i + 2][j] :
            score += 1
            
    
    return score, (i,j)
    
    




def solve_board(board):

    #iterate board
    for i in range(6, -1 , -1):
        for j in range(7, -1, -1):

            
            if board[i][j] == 'riya':
                continue
            #blue green red
            if board[i][j] == 'yellow' or board[i][j] == 'red' or board[i][j] == 'blue' or board[i][j] == 'green':
                #check borader
                #left
                if j > 0 and board[i][j - 1] != 'riya':
                    #new board
                    new_board = np.copy(board)
                    #move
                    tmp = new_board[i][j]
                    new_board[i][j] = new_board[i][j - 1]
                    new_board[i][j - 1] = tmp
                    
                    #solve
                    result = solve(new_board, i, j - 1)
                    if result is not None:
                        return board[i][j],[(i,j),result]
                #top
                if i > 0 and board[i - 1][j] != 'riya':
                    #new board
                    new_board = np.copy(board)
                    #move
                    tmp = new_board[i][j]
                    new_board[i][j] = new_board[i - 1][j]
                    new_board[i - 1][j] = tmp
                    
                    #solve
                    result = solve(new_board, i - 1, j)
                    if result is not None:
                        return board[i][j],[(i,j),result]
                #right
                if j < 7 and board[i][j + 1] != 'riya':
                    #new board
                    new_board = np.copy(board)
                    #move
                    tmp = new_board[i][j]
                    new_board[i][j] = new_board[i][j + 1]
                    new_board[i][j + 1] = tmp
                    
                    #solve
                    result = solve(new_board, i, j + 1)
                    if result is not None:
                        return board[i][j],[(i,j),result]
                #bottom
                if i < 6 and board[i + 1][j ] != 'riya':
                    #new board
                    new_board = np.copy(board)
                    #move
                    tmp = new_board[i][j]
                    new_board[i][j] = new_board[i + 1][j]
                    new_board[i + 1][j] = tmp
                    
                    #solve
                    result = solve(new_board, i + 1, j)
                    if result is not None:
                        return board[i][j],[(i,j),result]
                    
                if board[i][j] == 'purple':
                    return board[i][j], [(i,j),(0,0)]
                    # continue
    # for i in range(6, -1 , -1):
    #     for j in range(7, -1, -1):
    #         if board[i][j] == 'purple':
    #                 return board[i][j], [(i,j),(0,0)]
    return None, None

def solve_board_order(board):
    #iterate order
    for gtype in ['red','blue', 'green', 'yellow','purple']:
        #iterate board
        for i in range(6, -1 , -1):
            for j in range(7, -1, -1):
                if board[i][j] == gtype:
                
                    if board[i][j] == 'riya':
                        continue
                    #blue green red
                    if board[i][j] == 'yellow' or board[i][j] == 'red' or board[i][j] == 'blue' or board[i][j] == 'green':
                        #check borader
                        #left
                        if j > 0 and board[i][j - 1] != 'riya':
                            #new board
                            new_board = np.copy(board)
                            #move
                            tmp = new_board[i][j]
                            new_board[i][j] = new_board[i][j - 1]
                            new_board[i][j - 1] = tmp
                            
                            #solve
                            result = solve(new_board, i, j - 1)
                            if result is not None:
                                return board[i][j],[(i,j),result]
                        #top
                        if i > 0 and board[i - 1][j] != 'riya':
                            #new board
                            new_board = np.copy(board)
                            #move
                            tmp = new_board[i][j]
                            new_board[i][j] = new_board[i - 1][j]
                            new_board[i - 1][j] = tmp
                            
                            #solve
                            result = solve(new_board, i - 1, j)
                            if result is not None:
                                return board[i][j],[(i,j),result]
                        #right
                        if j < 7 and board[i][j + 1] != 'riya':
                            #new board
                            new_board = np.copy(board)
                            #move
                            tmp = new_board[i][j]
                            new_board[i][j] = new_board[i][j + 1]
                            new_board[i][j + 1] = tmp
                            
                            #solve
                            result = solve(new_board, i, j + 1)
                            if result is not None:
                                return board[i][j],[(i,j),result]
                        #bottom
                        if i < 6 and board[i + 1][j ] != 'riya':
                            #new board
                            new_board = np.copy(board)
                            #move
                            tmp = new_board[i][j]
                            new_board[i][j] = new_board[i + 1][j]
                            new_board[i + 1][j] = tmp
                            
                            #solve
                            result = solve(new_board, i + 1, j)
                            if result is not None:
                                return board[i][j],[(i,j),result]
                        if board[i][j] == 'purple':
                            return board[i][j], [(i,j),(0,0)]
    return None, None
        
def solve(board, i, j):
    #horizontal
    #center
    if j > 0 and j < 7:
        if board[i][j] == board[i][j - 1] and board[i][j] == board[i][j + 1]:
            return (i,j)
    #left
    if j > 1:
        if board[i][j] == board[i][j - 1] and board[i][j] == board[i][j - 2]:
            return (i,j)
    #right
    if j < 6:
        if board[i][j] == board[i][j + 1] and board[i][j] == board[i][j + 2]:
            return (i,j)
        
    #vertical
    #center
    if i > 0 and i < 6:
        if board[i][j] == board[i - 1][j] and board[i][j] == board[i + 1][j] :
            return (i,j)
    #top
    if i > 1:
        if board[i][j] == board[i - 1][j] and board[i][j] == board[i - 2][j] :
            return (i,j)
    #bottom
    if i < 5:
        if board[i][j] == board[i + 1][j] and board[i][j] == board[i + 2][j] :
            return (i,j)
    return None

def solve_normal(result,width,height,grids):
    img_board = np.array(grids).reshape((-1,2))
    print(result[0][1])
    start_point = (int(img_board[result[0][0]*8 + result[0][1]][0] + width/2) , int(img_board[result[0][0]*8 + result[0][1]][1] + height / 2 ))
    end_point = (int(img_board[result[1][0]*8 + result[1][1]][0] + width/2) , int(img_board[result[1][0]*8 + result[1][1]][1] + height / 2 ))
    # print(img_board)
    # start_point = [int(img_board[result[0][0]][result[0][1]][0] + width / 2) ,int(img_board[result[0][0]][result[0][1]][1] + height / 2 )]
    # end_point = [int(img_board[result[1][0]][result[1][1]][0] + width / 2) , int(img_board[result[1][0]][result[1][1]][1] + height / 2)]
    print(start_point,end_point)
    hwnd.MouseSlide(start_point, end_point,speed=5)
    # hwnd.MouseMoveCenter()
    


    # hwnd.MouseMoveCenter()

def exacute_game(gtype, result , width,height,grids):
    if gtype == 'riya':
        pass
    
    elif gtype == 'purple':
        img_board = np.array(grids).reshape((-1,2))
        start_point = (int(img_board[result[0][0]*8 + result[0][1]][0] + width/2) , int(img_board[result[0][0]*8 + result[0][1]][1] + height / 2 ))
        for i in range(3):
            hwnd.MouseRighttClickPos(start_point)
    
    
    elif gtype == 'yellow' or gtype == 'blue' or gtype == 'green' or gtype == 'red':
        solve_normal(result,width,height,grids)
        
        time.sleep(3) # wait animation
     
    item1 = (390,675)
    item2 = (442,675)
    item3 = (510,675)
    item4 = (566,675)
    
    # time.sleep(5)
    for i in range(10):
        hwnd.MouseLefttClickPos(item4)
        time.sleep(0.1)
    for i in range(10):
        hwnd.MouseLefttClickPos(item3)
        time.sleep(0.1)
    for i in range(1):
        hwnd.MouseLefttClickPos(item2)
        time.sleep(0.1)
    for i in range(1):
        hwnd.MouseLefttClickPos(item1)
        time.sleep(0.1)
    
    


""" END """


if __name__ == '__main__':
    # p = Process(target=preview, args=[hwnd,])
    # p2 = Process(target=shoot, args=[hwnd,])
    # # p.start()
    # p2.start()
    # # p.join()
    # p2.join()
    # preview(hwnd)
    # capture_save(hwnd)
    # im_path = os.path.join(mirror_save_path, '40.jpg')
    # im = cv2.imread(im_path)
    retry = 0
    while True:
        try:
            hwnd.ResetWindow()
            im = hwnd.CaptureCV(method='mss', wtype='universal')
            # width,height,grids = preprocess_img(im)
            width,height,grids = preprocess_shit()
            board = generate_board(width,height, grids,im)
            
            print(board)
            
            gtype, result = solve_board_optimal(board)
            
            if gtype is None :
                gtype, result = solve_board(board)
            
            if gtype:
                exacute_game(gtype, result, width, height, grids)
            else :
                retry += 1
                if retry > 20:
                    quit(0)
                for i in range(3):
                    time.sleep(1)
                    logging.info('Sleep '+ str(i+1))
            
         
        except Exception as e:
            print(e)
            raise e
            # for i in range(3):
            #     time.sleep(1)
            #     logging.info('Sleep '+ str(i+1))
            # pass
    
    logging.info('Complete')

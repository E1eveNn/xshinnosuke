import os
import time
import math


class Robot:
    def __init__(self, _board):
        self.board = _board

    def haveValuePoints(self, player, enemy, board):
        points = []

        for x in range(15):
            for y in range(15):
                list1 = []
                list2 = []
                list3 = []
                list4 = []
                if self.board[x][y] == -1:
                    for tmp in range(9):
                        i = x + tmp - 4
                        j = y + tmp - 4
                        if i < 0 or i > 14:
                            list1.append(-2)
                        else:
                            list1.append(board[i][y])
                        if j < 0 or j > 14:
                            list2.append(-2)
                        else:
                            list2.append(board[x][j])
                        if i < 0 or j < 0 or i > 14 or j > 14:
                            list3.append(-2)
                        else:
                            list3.append(board[i][j])
                        k = y - tmp + 4
                        if i < 0 or k < 0 or i > 14 or k > 14:
                            list4.append(-2)
                        else:
                            list4.append(board[i][k])


                    playerValue = self.value_point(player, enemy, list1, list2, list3, list4)
                    enemyValue = self.value_point(enemy, player, list1, list2, list3, list4)
                    if enemyValue >= 10000:
                        enemyValue -= 500
                    elif enemyValue >= 5000:
                        enemyValue -= 300
                    elif enemyValue >= 2000:
                        enemyValue -= 250
                    elif enemyValue >= 1500:
                        enemyValue -= 200
                    elif enemyValue >= 99:
                        enemyValue -= 10
                    elif enemyValue >= 5:
                        enemyValue -= 1
                    value = playerValue + enemyValue
                    if value > 0:
                        points.append([x, y, value])
        return points

    def MaxValue_po(self, player, enemy):
        points = self.haveValuePoints(player, enemy, self.board)
        flag = 0
        _point = []
        for p in points:
            if p[2] > flag:
                _point = p
                flag = p[2]
        return _point[0], _point[1], _point[2]

    def value_point(self, player, enemy, list1, list2, list3, list4):

        flag = 0
        flag += self.willbefive(player, list1)
        flag += self.willbefive(player, list2)
        flag += self.willbefive(player, list3)
        flag += self.willbefive(player, list4)
        flag += self.willbealive4(player, list1)
        flag += self.willbealive4(player, list2)
        flag += self.willbealive4(player, list3)
        flag += self.willbealive4(player, list4)
        flag += self.willbesleep4(player, enemy, list1)
        flag += self.willbesleep4(player, enemy, list2)
        flag += self.willbesleep4(player, enemy, list3)
        flag += self.willbesleep4(player, enemy, list4)
        flag += self.willbealive3(player, list1)
        flag += self.willbealive3(player, list2)
        flag += self.willbealive3(player, list3)
        flag += self.willbealive3(player, list4)
        flag += self.willbesleep3(player, enemy, list1)
        flag += self.willbesleep3(player, enemy, list2)
        flag += self.willbesleep3(player, enemy, list3)
        flag += self.willbesleep3(player, enemy, list4)
        flag += self.willbealive2(player, enemy, list1)
        flag += self.willbealive2(player, enemy, list2)
        flag += self.willbealive2(player, enemy, list3)
        flag += self.willbealive2(player, enemy, list4)
        flag += self.willbesleep2(player, enemy, list1)
        flag += self.willbesleep2(player, enemy, list2)
        flag += self.willbesleep2(player, enemy, list3)
        flag += self.willbesleep2(player, enemy, list4)
        return flag

    @staticmethod
    def willbefive(player, checklist):
        if checklist[0] == player and checklist[1] == player and \
                checklist[2] == player and checklist[3] == player:
            return 10000
        elif checklist[5] == player and checklist[6] == player and \
                checklist[7] == player and checklist[8] == player:
            return 10000
        elif checklist[2] == player and checklist[3] == player and \
                checklist[5] == player and checklist[6] == player:
            return 10000
        elif checklist[1] == player and checklist[2] == player and \
                checklist[3] == player and checklist[5] == player:
            return 10000
        elif checklist[3] == player and checklist[5] == player and \
                checklist[6] == player and checklist[7] == player:
            return 10000
        else:
            return 0

    @staticmethod
    def willbealive4(player, checklist):
        if checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == -1:
            return 5000
        elif checklist[3] == -1 and checklist[5] == player and \
                checklist[6] == player and checklist[7] == player \
                and checklist[8] == -1:
            return 5000
        elif checklist[1] == -1 and checklist[2] == player and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == -1:
            return 5000
        elif checklist[2] == -1 and checklist[3] == player and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == -1:
            return 5000
        else:
            return 0

    @staticmethod
    def willbesleep4(player, enemy, checklist):
        if checklist[0] == enemy and checklist[1] == player and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == -1:
            return 1700
        elif checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == -1:
            return 1700
        elif checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == -1:
            return 1700
        elif checklist[3] == enemy and checklist[5] == player and \
                checklist[6] == player and checklist[7] == player \
                and checklist[8] == -1:
            return 1700
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == enemy:
            return 1700
        elif checklist[1] == -1 and checklist[2] == player and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == enemy:
            return 1700
        elif checklist[2] == -1 and checklist[3] == player and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == enemy:
            return 1700
        elif checklist[3] == -1 and checklist[5] == player and \
                checklist[6] == player and checklist[7] == player \
                and checklist[8] == enemy:
            return 1700
        else:
            return 0

    @staticmethod
    def willbealive3(player, checklist):
        if checklist[0] == -1 and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == -1:
            return 1900
        elif checklist[1] == -1 and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == -1:
            return 1900
        elif checklist[2] == -1 and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == -1:
            return 1900
        elif checklist[1] == -1 and checklist[2] == player and \
                checklist[3] == player and checklist[5] == -1 \
                and checklist[6] == -1:
            return 1900
        elif checklist[2] == -1 and checklist[3] == player and \
                checklist[5] == player and checklist[6] == -1 \
                and checklist[7] == -1:
            return 1900
        elif checklist[3] == -1 and checklist[5] == player and \
                checklist[6] == player and checklist[7] == -1 \
                and checklist[8] == -1:
            return 1900
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == -1:
            return 1600
        elif checklist[2] == -1 and checklist[3] == player and \
                checklist[6] == player and checklist[5] == -1 \
                and checklist[7] == -1:
            return 1600
        elif checklist[3] == -1 and checklist[5] == player and \
                checklist[7] == player and checklist[6] == -1 \
                and checklist[8] == -1:
            return 1600
        elif checklist[3] == -1 and checklist[5] == -1 and \
                checklist[7] == player and checklist[6] == player \
                and checklist[8] == -1:
            return 1600
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[6] == -1:
            return 1600
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[6] == -1:
            return 1600
        else:
            return 0

    @staticmethod
    def willbesleep3(player, enemy, checklist):
        if checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == player and checklist[5] == -1 \
                and checklist[6] == -1:
            return 350
        elif checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == player and checklist[6] == -1 \
                and checklist[7] == -1:
            return 350
        elif checklist[3] == enemy and checklist[5] == player and \
                checklist[6] == player and checklist[7] == -1 \
                and checklist[8] == -1:
            return 350
        elif checklist[0] == -1 and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == enemy:
            return 350
        elif checklist[1] == -1 and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == enemy:
            return 350
        elif checklist[2] == -1 and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == enemy:
            return 350
        elif checklist[0] == enemy and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == player \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 300
        elif checklist[1] == enemy and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == player \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 300
        elif checklist[2] == enemy and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == player \
                and checklist[7] == -1 and checklist[8] == enemy:
            return 300
        elif checklist[0] == enemy and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 300
        elif checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 300
        elif checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == -1 and checklist[6] == player \
                and checklist[7] == -1 and checklist[8] == enemy:
            return 300
        elif checklist[0] == enemy and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 300
        elif checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 300
        elif checklist[3] == enemy and checklist[5] == -1 and \
                checklist[6] == player and checklist[7] == player \
                and checklist[8] == -1:
            return 300
        elif checklist[0] == enemy and checklist[1] == player and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == -1:
            return 300
        elif checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == -1 and checklist[6] == player \
                and checklist[7] == -1:
            return 300
        elif checklist[3] == enemy and checklist[5] == player and \
                checklist[6] == -1 and checklist[7] == player \
                and checklist[8] == -1:
            return 300
        elif checklist[0] == player and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == -1 \
                and checklist[5] == enemy:
            return 300
        elif checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == -1 and checklist[6] == -1 \
                and checklist[7] == player:
            return 300
        elif checklist[3] == enemy and checklist[5] == player and \
                checklist[6] == -1 and checklist[7] == -1 \
                and checklist[8] == player:
            return 300
        elif checklist[0] == player and checklist[1] == -1 and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == enemy:
            return 300
        elif checklist[1] == player and checklist[2] == -1 and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == enemy:
            return 300
        elif checklist[3] == enemy and checklist[5] == -1 and \
                checklist[6] == -1 and checklist[7] == player \
                and checklist[8] == player:
            return 300
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == enemy:
            return 30
        elif checklist[2] == -1 and checklist[3] == player and \
                checklist[5] == -1 and checklist[6] == player \
                and checklist[7] == enemy:
            return 300
        elif checklist[3] == -1 and checklist[5] == player and \
                checklist[6] == -1 and checklist[7] == player \
                and checklist[8] == enemy:
            return 300
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == enemy:
            return 300
        elif checklist[1] == -1 and checklist[2] == player and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == enemy:
            return 300
        elif checklist[3] == -1 and checklist[5] == -1 and \
                checklist[6] == player and checklist[7] == player \
                and checklist[8] == enemy:
            return 300
        elif checklist[0] == player and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == enemy:
            return 300
        elif checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == -1 and checklist[5] == -1 \
                and checklist[6] == player:
            return 300
        elif checklist[2] == player and checklist[3] == -1 and \
                checklist[5]== -1 and checklist[6] == player \
                and checklist[7] == enemy:
            return 300
        elif checklist[3] == enemy and checklist[5] == -1 and \
                checklist[6] == player and checklist[7] == -1 \
                and checklist[8] == player:
            return 300
        else:
            return 0

    @staticmethod
    def willbealive2(player, enemy, checklist):
        if checklist[1] == -1 and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == -1 \
                and checklist[6] == -1:
            return 99
        elif checklist[2] == -1 and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == -1 \
                and checklist[7] == -1:
            return 99
        elif checklist[0] == -1 and checklist[1] == -1 and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 99
        elif checklist[1] == -1 and checklist[2] == -1 and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 99
        elif checklist[1] == enemy and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == -1 \
                and checklist[6] == -1 and checklist[7] == -1:
            return 99
        elif checklist[2] == enemy and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == -1 \
                and checklist[7] == -1 and checklist[8] == -1:
            return 99
        else:
            return 0

    @staticmethod
    def willbesleep2(player, enemy, checklist):
        if checklist[2] == enemy and checklist[3] == player and \
                checklist[5] == -1 and checklist[6] == -1 \
                and checklist[7] == -1:
            return 5
        elif checklist[3] == enemy and checklist[5] == player and \
                checklist[6] == -1 and checklist[7] == -1 \
                and checklist[8] == -1:
            return 5
        elif checklist[0] == -1 and checklist[1] == -1 and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == enemy:
            return 5
        elif checklist[1] == -1 and checklist[2] == -1 and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == enemy:
            return 5
        elif checklist[1] == enemy and checklist[2] == -1 and \
                checklist[3] == player and checklist[5] == -1 \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 5
        elif checklist[2] == enemy and checklist[3] == -1 and \
                checklist[5] == player and checklist[6] == -1 \
                and checklist[7] == -1 and checklist[8] == enemy:
            return 5
        elif checklist[0] == enemy and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 5
        elif checklist[2] == enemy and checklist[3] == -1 and \
                checklist[5] == -1 and checklist[6] == player \
                and checklist[7] == -1 and checklist[8] == enemy:
            return 5
        elif checklist[0] == enemy and checklist[1] == -1 and \
                checklist[2] == -1 and checklist[3] == player \
                and checklist[5] == -1 and checklist[6] == enemy:
            return 5
        elif checklist[1] == enemy and checklist[2] == -1 and \
                checklist[3] == -1 and checklist[5] == player \
                and checklist[6] == -1 and checklist[7] == enemy:
            return 5
        elif checklist[0] == -1 and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == -1 \
                and checklist[5] == enemy:
            return 5
        elif checklist[3] == -1 and checklist[5] == -1 and \
                checklist[6] == -1 and checklist[7] == player \
                and checklist[8] == enemy:
            return 5
        elif checklist[0] == -1 and checklist[1] == -1 and \
                checklist[2] == player and checklist[3] == -1 \
                and checklist[5] == enemy:
            return 5
        elif checklist[2] == -1 and checklist[3] == -1 and \
                checklist[5] == -1 and checklist[6] == player \
                and checklist[7] == enemy:
            return 5
        elif checklist[1] == enemy and checklist[2] == player and \
                checklist[3] == -1 and checklist[5] == -1 \
                and checklist[6] == -1:
            return 5
        elif checklist[3] == enemy and checklist[5] == -1 and \
                checklist[6] == player and checklist[7] == -1 \
                and checklist[8] == -1:
            return 5
        elif checklist[0] == enemy and checklist[1] == player and \
                checklist[2] == -1 and checklist[3] == -1 \
                and checklist[5] == -1:
            return 5
        elif checklist[3] == enemy and checklist[5] == -1 and \
                checklist[6] == -1 and checklist[7] == player \
                and checklist[8] == -1:
            return 5
        else:
            return 0


class SGFflie:
    def __init__(self, save_path='./save/', train_path='./train'):
        self.POS = 'abcdefghijklmno'
        self.savepath = save_path
        self.trainpath = train_path

    def openfile(self, filepath):
        f = open(filepath, 'r')
        data = f.read()
        f.close()

        effective_data = data.split(';')
        s = effective_data[2:-1]

        board = []
        step = 0
        for point in s:
            x = self.POS.index(point[2])
            y = self.POS.index(point[3])
            color = step % 2
            step += 1
            board.append([x, y, color, step])

        return board

    def savefile(self, board):
        data = self.createdata(board)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        filepath = os.path.join(self.savepath, data.split(';')[1] + ".sgf")
        f = open(filepath, 'w')
        f.write(data)
        f.close()
        return

    def createdata(self, board):
        now = time.localtime(time.time())
        _time = ''
        for index in range(6):
            _time = _time + str(now[index])
        data = '(;' + _time + ";"

        for it in board:
            if it[2] == 0:
                data = data + 'B[' + self.POS[it[0]] + self.POS[it[1]] + "];"
            else:
                data = data + 'W[' + self.POS[it[0]] + self.POS[it[1]] + "];"
        data = data + ')'
        return data

    def createTraindataFromqipu(self, path, color=0):
        qipu = self.openfile(path)

        bla = qipu[::2]
        whi = qipu[1::2]
        bla_step = len(bla)
        whi_step = len(whi)

        train_x = []
        train_y = []

        if color == 0:
            temp_x = [0.0 for i in range(225)]
            for index in range(bla_step):
                _x = [0.0 for i in range(225)]
                _y = [0.0 for i in range(225)]
                if index == 0:
                    train_x.append(_x)
                    _y[bla[index][0]*15 + bla[index][1]] = 2.0
                    train_y.append(_y)
                else:
                    _x = temp_x.copy()
                    train_x.append(_x)
                    _y[bla[index][0] * 15 + bla[index][1]] = 2.0
                    train_y.append(_y)

                temp_x[bla[index][0] * 15 + bla[index][1]] = 2.0
                if index < whi_step:
                    temp_x[whi[index][0] * 15 + whi[index][1]] = 1.0
        return train_x, train_y

    def createTraindataFromqipu1(self, path, color=0):
        qipu = self.openfile(path)

        bla = qipu[::2]
        whi = qipu[1::2]
        bla_step = len(bla)
        whi_step = len(whi)

        train_x = []
        train_y = []

        if color == 0:
            temp_x = [[[0.0, 0.0, 0.0] for j in range(15)] for k in range(15)]
            for index in range(bla_step):
                _x = [[[0.0, 0.0, 0.0] for j in range(15)] for k in range(15)]
                _y = [0.0 for i in range(225)]
                if index == 0:
                    train_x.append(_x)
                    _y[bla[index][0]*15 + bla[index][1]] = 1.0
                    train_y.append(_y)
                else:
                    _x = temp_x.copy()
                    train_x.append(_x)
                    _y[bla[index][0] * 15 + bla[index][1]] = 1.0
                    train_y.append(_y)

                temp_x[bla[index][0]][bla[index][1]][1] = 1.0
                if index < whi_step:
                    temp_x[whi[index][0]][whi[index][1]][2] = 1.0
        for tmp in train_x:
            for x in tmp:
                for y in x:
                    if y[1] == 0 and y[2] == 0:
                        y[0] = 1
        return train_x, train_y

    def createTraindata(self):
        filepath = self.allFileFromDir(self.savepath)
        train_x = []
        train_y = []
        for path in filepath:
            x, y = self.createTraindataFromqipu(path)
            train_x = train_x + x
            train_y = train_y + y
        return train_x, train_y

    @staticmethod
    def allFileFromDir(Dirpath):
        pathDir = os.listdir(Dirpath)
        pathfile = []
        for allDir in pathDir:
            child = os.path.join('%s%s' % (Dirpath, allDir))
            pathfile.append(child)
        return pathfile

    def createqijuFromqipu(self, path):
        qipu = self.openfile(path)
        bla = qipu[::2]
        whi = qipu[1::2]
        qiju = [[-1]*15 for i in range(15)]

        for tmp in bla:
            qiju[tmp[0]][tmp[1]] = -2
        for tmp in whi:
            qiju[tmp[0]][tmp[1]] = -7
        return qiju


def get_distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def pos_in_board(x, y):
    return x * 30 + 25, y * 30 + 25


def pos_in_qiju(x, y):
    return int((x - 25) / 30), int((y - 25) / 30)


def pos_to_draw(*args):
    x, y = args
    return x - 11, y - 11, x + 11, y + 11


def click_in_board(x, y):
    return 10 < x < 460 and 10 < y < 460

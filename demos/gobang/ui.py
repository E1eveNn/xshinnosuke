#coding=utf-8
'''
这是主程序，主要定义了Gobang和Robot两个对象：
1.Gobang主要是图像界面交互的实现。
2.Robot是基于五子棋的一些基本规则写出来的一个简单智能程序，不包含
神经网络的搭建。
'''

from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile
from .network import CNN
from .utils import *


class GoBang:
    def __init__(self):
        self.someoneWin = False
        self.humanChessed = False
        self.IsStart = False
        self.player = 0
        self.playmethod = 0
        self.bla_start_pos = [235, 235]
        self.whi_chessed = []
        self.bla_chessed = []
        self.board = self.init_board()
        self.window = Tk()
        self.var = IntVar()
        self.var.set(0)
        self.var1 = IntVar()
        self.var1.set(0)
        self.window.title("myGoBang")
        self.window.geometry("600x470+80+80")
        self.window.resizable(0, 0)
        self.can = Canvas(self.window, bg="#EEE8AC", width=470, height=470)
        self.draw_board()
        self.can.grid(row=0, column=0)
        self.net_board = self.get_net_board()
        self.robot = Robot(self.board)
        self.sgf = SGFflie()
        self.net = CNN()

    @staticmethod
    def init_board():
        list1 = [[-1]*15 for i in range(15)]
        return list1

    def draw_board(self):
        for row in range(15):
            if row == 0 or row == 14:
                self.can.create_line((25, 25 + row * 30), (445, 25 + row * 30), width=2)
            else:
                self.can.create_line((25, 25 + row * 30), (445, 25 + row * 30), width=1)
        for col in range(15):
            if col == 0 or col == 14:
                self.can.create_line((25 + col * 30, 25), (25 + col * 30, 445), width=2)
            else:
                self.can.create_line((25 + col * 30, 25), (25 + col * 30, 445), width=1)
        self.can.create_oval(112, 112, 118, 118, fill="black")
        self.can.create_oval(352, 112, 358, 118, fill="black")
        self.can.create_oval(112, 352, 118, 358, fill="black")
        self.can.create_oval(232, 232, 238, 238, fill="black")
        self.can.create_oval(352, 352, 358, 358, fill="black")

    def get_nearest_po(self, x, y):
        flag = 600
        position = ()
        for point in self.net_board:
            distance = get_distance([x, y], point)
            if distance < flag:
                flag = distance
                position = point
        return position

    def no_in_chessed(self, pos):
        whi_chess = self.check_chessed(pos, self.whi_chessed)
        bla_chess = self.check_chessed(pos, self.bla_chessed)
        return whi_chess == False and bla_chess == False

    def ai_no_in_chessed(self, pos, value):
        no_in_chessed = self.no_in_chessed(pos)
        return no_in_chessed and value < 4000

    @staticmethod
    def check_chessed(point, chessed):
        if len(chessed) == 0:
            return False
        flag = 0
        for p in chessed:
            if point[0] == p[0] and point[1] == p[1]:
                flag = 1
        if flag == 1:
            return True
        else:
            return False

    def have_five(self, chessed):
        if len(chessed) == 0:
            return False
        for row in range(15):
            for col in range(15):
                x = 25 + row * 30
                y = 25 + col * 30
                if self.check_chessed((x, y), chessed) and \
                                self.check_chessed((x, y + 30), chessed) and \
                                self.check_chessed((x, y + 60), chessed) and \
                                self.check_chessed((x, y + 90), chessed) and \
                                self.check_chessed((x, y + 120), chessed):
                    return True
                elif self.check_chessed((x, y), chessed) and \
                                self.check_chessed((x + 30, y), chessed) and \
                                self.check_chessed((x + 60, y), chessed) and \
                                self.check_chessed((x + 90, y), chessed) and \
                                self.check_chessed((x + 120, y), chessed):
                    return True
                elif self.check_chessed((x, y), chessed) and \
                                self.check_chessed((x + 30, y + 30), chessed) and \
                                self.check_chessed((x + 60, y + 60), chessed) and \
                                self.check_chessed((x + 90, y + 90), chessed) and \
                                self.check_chessed((x + 120, y + 120), chessed):
                    return True
                elif self.check_chessed((x, y), chessed) and \
                                self.check_chessed((x + 30, y - 30), chessed) and \
                                self.check_chessed((x + 60, y - 60), chessed) and \
                                self.check_chessed((x + 90, y - 90), chessed) and \
                                self.check_chessed((x + 120, y - 120), chessed):
                    return True
                else:
                    pass
        return False

    def check_win(self):
        if self.have_five(self.whi_chessed):
            label = Label(self.window, text="White Win!", background='#FFF8DC', font=("yahei", 15, "bold"))
            label.place(relx=0, rely=0, x=480, y=40)
            return True
        elif self.have_five(self.bla_chessed):
            label = Label(self.window, text="Black Win!", background='#FFF8DC', font=("yahei", 15, "bold"))
            label.place(relx=0, rely=0, x=480, y=40)
            return True
        else:
            return False

    def draw_chessed(self):
        if len(self.whi_chessed) != 0:
            for tmp in self.whi_chessed:
                oval = pos_to_draw(*tmp[0:2])
                self.can.create_oval(oval, fill="white")

        if len(self.bla_chessed) != 0:
            for tmp in self.bla_chessed:
                oval = pos_to_draw(*tmp[0:2])
                self.can.create_oval(oval, fill="black")

    def draw_a_chess(self, x, y, player=None):
        _x, _y = pos_in_qiju(x, y)
        oval = pos_to_draw(x, y)

        if player == 0:
            self.can.create_oval(oval, fill="black")
            self.bla_chessed.append([x, y, 0])
            self.board[_x][_y] = 1
        elif player == 1:
            self.can.create_oval(oval, fill="white")
            self.whi_chessed.append([x, y, 1])
            self.board[_x][_y] = 0
        else:
            print(AttributeError("please select player"))
        return

    def AIrobotChess(self):
        cnn_predict = self.net.prediction(self.board)
        if self.player % 2 == 0:
            if len(self.bla_chessed) == 0 and len(self.whi_chessed) == 0:
                self.draw_a_chess(*self.bla_start_pos, 0)
            else:
                _x, _y, _ = self.robot.MaxValue_po(1, 0)
                newPoint = pos_in_board(_x, _y)
                if self.ai_no_in_chessed(cnn_predict, _):
                    self.draw_a_chess(*cnn_predict, 0)
                else:
                    self.draw_a_chess(*newPoint, 0)

        else:
            self.robotChess()

    def robotChess(self):
        if self.player == 0:
            if len(self.bla_chessed) == 0 and len(self.whi_chessed) == 0:
                self.draw_a_chess(*self.bla_start_pos, player=0)
                return
            else:
                _x, _y, _ = self.robot.MaxValue_po(0, 1)
                newPoint = pos_in_board(_x, _y)
                self.draw_a_chess(*newPoint, player=0)
        else:#白棋下
            _x, _y, _ = self.robot.MaxValue_po(1, 0)
            newPoint = pos_in_board(_x, _y)
            self.draw_a_chess(*newPoint, player=1)

    def chess(self, event):
        if self.someoneWin is True or self.IsStart is False:
            return

        ex = event.x
        ey = event.y
        if not click_in_board(ex, ey):
            return

        neibor_po = self.get_nearest_po(ex, ey)
        if self.no_in_chessed(neibor_po):
            if self.player == 3:
                total_chessed = len(self.bla_chessed)\
                                + len(self.whi_chessed)
                if (total_chessed % 2) == 0:
                    self.draw_a_chess(*neibor_po, 0)
                else:
                    self.draw_a_chess(*neibor_po, 1)
                self.someoneWin = self.check_win()
            else:
                if self.player == 0:
                    self.draw_a_chess(*neibor_po, 1)
                else:
                    self.draw_a_chess(*neibor_po, 0)

                self.someoneWin = self.check_win()
                if self.playmethod == 0:
                    self.AIrobotChess()
                else:
                    self.robotChess()
                self.someoneWin = self.check_win()

    @staticmethod
    def get_net_board():
        net_list = []
        for row in range(15):
            for col in range(15):
                point = pos_in_board(row, col)
                net_list.append(point)
        return net_list

    def resetButton(self):
        self.someoneWin = False
        self.IsStart = False
        self.whi_chessed.clear()
        self.bla_chessed.clear()
        self.board = self.init_board()
        self.robot = Robot(self.board)
        label = Label(self.window,
                      text="          ",
                      background="#F0F0F0",
                      font=("yahei", 15, "bold"))
        label.place(relx=0, rely=0, x=480, y=40)
        self.can.delete("all")
        self.draw_board()
        self.can.grid(row=0, column=0)

    def BakcAChess(self):
        if self.someoneWin == False:
            if len(self.whi_chessed) != 0:
                p = self.whi_chessed.pop()
                x, y = pos_in_qiju(*p[0:2])
                self.board[x][y] = -1

            if self.player == 0 and len(self.bla_chessed) != 1:
                p = self.bla_chessed.pop()
                x, y = pos_in_qiju(*p[0:2])
                self.board[x][y] = -1

            elif self.player == 1 and len(self.bla_chessed) != 0:
                p = self.bla_chessed.pop()
                x, y = pos_in_qiju(*p[0:2])
                self.board[x][y] = -1

            else:
                pass

            self.can.delete("all")
            self.draw_board()
            self.draw_chessed()

    def startButton(self):
        if not self.IsStart:
            self.IsStart = True
            if self.player % 2 == 0:
                if self.playmethod == 0:
                    self.AIrobotChess()
                elif self.playmethod == 1:
                    self.robotChess()
                self.draw_chessed()

    def selectColor(self):
        if not self.IsStart:
            if self.var.get() == 0:
                self.player = 0
            elif self.var.get() == 1:
                self.player = 1
            elif self.var.get() == 3:
                self.player = 3
            else:
                pass

        return

    def selectMathod(self):
        if self.IsStart == False:
            if self.var1.get() == 0:
                self.playmethod = 0
            elif self.var1.get() == 1:
                self.playmethod = 1
            else:
                pass
        return

    def createqipu(self):
        qipu = []
        step = 0
        totalstep = len(self.whi_chessed) + len(self.bla_chessed)
        while step < totalstep:
            if totalstep == 0:
                break
            flag = int(step / 2)
            if step % 2 == 0:
                pos = pos_in_qiju(*self.bla_chessed[flag][0:2])
                qipu.append([*pos, 0, step + 1])
            else:
                pos = pos_in_qiju(*self.whi_chessed[flag][0:2])
                qipu.append([*pos, 1, step + 1])
            step += 1
        return qipu

    def OpenFile(self):
        file_path = askopenfilename(filetypes=(('sgf file', '*.sgf'),
                                                    ('All File', '*.*')))
        if len(file_path) == 0:
            return

        qipu = self.sgf.openfile(file_path)

        self.whi_chessed.clear()
        self.bla_chessed.clear()

        for point in qipu:
            pos = pos_in_board(*point[0:2])

            if point[2] == 0:
                self.bla_chessed.append([*pos, 0])
            else:
                self.whi_chessed.append([*pos, 1])

        self.can.delete("all")
        self.draw_board()
        self.draw_chessed()

    def SaveFile(self, method=1):
        qipu = self.createqipu()
        if method == 0:
            try:
                file = asksaveasfile(filetypes=(('sgf file', '*.sgf'),
                                                ('All File', '*.*')))
                file.close()
            except AttributeError:
                return

            pathName = file.name
            newName = pathName + '.sgf'
            os.rename(pathName, newName)
            f = open(newName, 'w')
            data = self.sgf.createdata(qipu)
            f.write(data)
            f.close()

        elif method == 1:
            self.sgf.savefile(qipu)

    def start(self):
        b3 = Button(self.window, text="start", command=self.startButton)
        b3.place(relx=0, rely=0, x=495, y=100)

        b1 = Button(self.window, text="reset", command=self.resetButton)
        b1.place(relx=0, rely=0, x=495, y=150)

        b2 = Button(self.window, text="withdraw", command=self.BakcAChess)
        b2.place(relx=0, rely=0, x=495, y=200)

        b4 = Radiobutton(self.window, text="choose white", variable=self.var, value=0, command=self.selectColor)
        b4.place(relx=0, rely=0, x=495, y=250)

        b5 = Radiobutton(self.window, text="choose black", variable=self.var, value=1, command=self.selectColor)
        b5.place(relx=0, rely=0, x=495, y=280)

        b6 = Button(self.window, text="open chessbook", command=self.OpenFile)
        b6.place(relx=0, rely=0, x=495, y=400)

        b7 = Button(self.window, text="save chessbook", command=self.SaveFile)
        b7.place(relx=0, rely=0, x=495, y=430)

        b8 = Radiobutton(self.window, text="ai", variable=self.var1, value=0, command=self.selectMathod)
        b8.place(relx=0, rely=0, x=490, y=340)

        b9 = Radiobutton(self.window, text="normal", variable=self.var1, value=1, command=self.selectMathod)
        b9.place(relx=0, rely=0, x=490, y=370)

        b10 = Radiobutton(self.window, text="p2p", variable=self.var, value=3, command=self.selectColor)
        b10.place(relx=0, rely=0, x=495, y=310)

        self.can.bind("<Button-1>", lambda x: self.chess(x))
        self.window.mainloop()


def go():
    game = GoBang()
    game.start()
    del game

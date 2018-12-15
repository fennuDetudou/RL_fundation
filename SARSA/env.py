'''
强化学习环境定义
'''
import copy

MAP = \
    '''
.........
.  x    .
.   x o .
.       .
.........
'''

MAP=MAP.strip().split('\n')
MAP=[[c for c in lines] for lines in MAP]

move_X=[-1,1,0,0]
move_Y=[0,0,-1,1]

class Env(object):

    def __init__(self):
        self.map=copy.deepcopy(MAP)
        self.x=1
        self.y=1
        self.step=0
        self.total_reward=0
        self.is_end=0

    def interact(self,action):
        '''
        与环境进行交互
        :param action:
        :return:
        '''
        assert self.is_end==0
        new_x=self.x+move_X[action]
        new_y=self.y+move_Y[action]
        new_pos_char=self.map[new_x][new_y]
        self.step+=1
        #### 奖励函数
        if new_pos_char == '.':
            reward = 0
        elif new_pos_char == ' ':
            self.x = new_x
            self.y = new_y
            reward = 0
        elif new_pos_char == 'o':
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '
            self.is_end = 1
            reward = 100
        elif new_pos_char == 'x':
            self.x = new_x
            self.y = new_y
            self.map[new_x][new_y] = ' '
            reward = -5
        self.total_reward += reward
        return reward

    @property
    def state_num(self):

        rows=len(self.map)  # 地图的x轴长度
        cols=len(self.map[0]) # 地图的y轴长度
        return rows*cols

    @property
    def present_state(self):
        # 返回A的位置
        cols = len(self.map[0])
        return self.x * cols + self.y

    def print_map(self):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        # 重新整理成矩阵形式
        print('\n'.join([''.join([c for c in line]) for line in printed_map]))

    def print_map_with_reprint(self, output_list):
        printed_map = copy.deepcopy(self.map)
        printed_map[self.x][self.y] = 'A'
        printed_list = [''.join([c for c in line]) for line in printed_map]
        for i, line in enumerate(printed_list):
            output_list[i] = line





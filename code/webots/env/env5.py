import matplotlib.pyplot as plt

from controller import Robot, Camera, Field
from controller import Motor
from controller import DistanceSensor, PositionSensor, Supervisor, GPS
import os
import numpy as np
import math
import random

SPAWN_LIMIT = 0.38
WALL_LIMIT = 0.46
MOVING_BOX_COUNT = 4
BOX_COUNT = MOVING_BOX_COUNT
MOVING_BOX_AXES = ("x", "x", "y", "y")
MOVING_BOX_AMPLITUDE = 0.11
MOVING_BOX_PHASE_STEP = 0.001
MOVING_PATH_MIN_DISTANCE = 0.11
MIN_SPAWN_DISTANCE = 0.12
BALL_SPAWN_DISTANCE = 0.14
OBJECT_COUNT = 3

def is_place(x,y,object1,object2,min_distance=MIN_SPAWN_DISTANCE):
    if (x > SPAWN_LIMIT or x < -SPAWN_LIMIT or y < -SPAWN_LIMIT or y > SPAWN_LIMIT):
        return False

    for ob in object1:
        if ob is None:
            continue
        r=pow((x-ob[0])**2+(y-ob[1])**2,0.5)
        if(r<min_distance):
            return False

    for ob in object2:
        if ob is None:
            continue
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)

        if (r < min_distance):
            return False

    return True

def sample_free_position(object1, object2, min_distance=MIN_SPAWN_DISTANCE, max_attempts=2000):
    for _ in range(max_attempts):
        x = random.random() * (2 * SPAWN_LIMIT) - SPAWN_LIMIT
        y = random.random() * (2 * SPAWN_LIMIT) - SPAWN_LIMIT
        if is_place(x, y, object1=object1, object2=object2, min_distance=min_distance):
            return x, y

    raise RuntimeError("Failed to sample a non-overlapping position in env3.")

def wall_detection(x,y):
    if(x>WALL_LIMIT or x<-WALL_LIMIT or y>WALL_LIMIT or y<-WALL_LIMIT):
        return True
    else:
        return False


def box_detection(x,y,object):
    for ob in object:
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)
        if (r < 0.115):
            return True

    else:
        return False

def ball_detection(x,y,object):
    for ob in object:
        r = pow((x - ob[0]) ** 2 + (y - ob[1]) ** 2, 0.5)
        if (r < 0.065):
            return True
        print('ball', r)
    else:
        return False

class env():
    def __init__(self, seed, Max_step):
        super(env, self).__init__()

        self.robot = Supervisor()

        self.name = self.robot.getFromDef("aqua")
        # 鍒濆鍖栨満鍣ㄤ汉浼犳劅鍣ㄤ俊鎭?
        self.trans = self.name.getField("translation")
        self.rotation = self.name.getField("rotation")
        self.timestep = int(self.robot.getBasicTimeStep())
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.left_wheel_motor = self.robot.getDevice('left wheel motor')
        self.right_wheel_motor = self.robot.getDevice('right wheel motor')



        self.left_wheel_motor.setPosition(float('inf'))
        self.right_wheel_motor.setPosition(float('inf'))

        self.object1=[]
        self.object2 = []
        self.box=[]
        for i in range(1, BOX_COUNT + 1):
            self.box.append(self.robot.getFromDef("box" + str(i)))
        self.box_translations = [box.getField("translation") for box in self.box]
        self.moving_box_phase = 0.0
        self.moving_box_paths = []
        self.moving_box_phase_offsets = []
        self.objects = []
        for i in range(1, OBJECT_COUNT + 1):
            self.objects.append(self.robot.getFromDef("target" + str(i)))

        self.object_num = OBJECT_COUNT
        self.instruction_text = ""
        self.update_instruction_label()
        # 璁惧畾浠跨湡鍔犻€?
        self.robot.simulationSetMode(2)

        # 璁剧珛闅忔満绉嶅瓙
        self.seed = seed
        self.set_seed = self.robot.getFromDef('n')
        self.set_seed.getField("randomSeed").setSFInt32(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        print("==============", self.seed,"==============")
        # 璁剧疆閫熷害涓婇檺浠ュ強浠跨湡鏃堕棿
        self.speed = self.left_wheel_motor.max_velocity
        self.delay_time = 20
        self.max_step = Max_step
        self.now_step = 0
        self.x=0
        self.y=0
        self.set_car_target()
        self.robot.step(self.timestep)
        # 璁板綍璺濈


    def get_instruction(self):
        return ""

    def get_object_spec(self, object_index):
        return "red"

    def update_instruction_label(self):
        self.robot.setLabel(
            0,
            "",
            0.02,
            0.04,
            0.08,
            0xFFFFFF,
            0.0,
            "Arial"
        )

    def get_text_state(self):
        # Preserve the legacy observation shape without exposing text information.
        return np.zeros((1, 6), dtype=np.float32)

    def sample_instruction(self):
        self.instruction_text = ""
        self.update_instruction_label()

    @staticmethod
    def moving_path_bounds(path):
        axis, center_x, center_y = path
        if axis == "x":
            return (
                center_x - MOVING_BOX_AMPLITUDE,
                center_x + MOVING_BOX_AMPLITUDE,
                center_y,
                center_y,
            )
        return (
            center_x,
            center_x,
            center_y - MOVING_BOX_AMPLITUDE,
            center_y + MOVING_BOX_AMPLITUDE,
        )

    @staticmethod
    def path_distance(path_a, path_b):
        ax0, ax1, ay0, ay1 = env.moving_path_bounds(path_a)
        bx0, bx1, by0, by1 = env.moving_path_bounds(path_b)
        dx = max(ax0 - bx1, bx0 - ax1, 0.0)
        dy = max(ay0 - by1, by0 - ay1, 0.0)
        return math.hypot(dx, dy)

    @staticmethod
    def point_to_path_distance(x, y, path):
        x0, x1, y0, y1 = env.moving_path_bounds(path)
        closest_x = min(max(x, x0), x1)
        closest_y = min(max(y, y0), y1)
        return math.hypot(x - closest_x, y - closest_y)

    def sample_moving_obstacle_layout(self):
        moving_center_limit = SPAWN_LIMIT - MOVING_BOX_AMPLITUDE
        paths = None
        for _ in range(100):
            candidate_paths = []
            for axis in MOVING_BOX_AXES:
                for _ in range(2000):
                    if axis == "x":
                        center_x = random.uniform(
                            -moving_center_limit,
                            moving_center_limit,
                        )
                        center_y = random.uniform(-SPAWN_LIMIT, SPAWN_LIMIT)
                    else:
                        center_x = random.uniform(-SPAWN_LIMIT, SPAWN_LIMIT)
                        center_y = random.uniform(
                            -moving_center_limit,
                            moving_center_limit,
                        )

                    candidate = (axis, center_x, center_y)
                    if all(
                        self.path_distance(candidate, existing)
                        >= MOVING_PATH_MIN_DISTANCE
                        for existing in candidate_paths
                    ):
                        candidate_paths.append(candidate)
                        break
                else:
                    break

            if len(candidate_paths) == MOVING_BOX_COUNT:
                paths = candidate_paths
                break

        if paths is None:
            raise RuntimeError(
                "Failed to sample non-overlapping moving obstacle paths."
            )

        self.moving_box_paths = paths
        self.moving_box_phase_offsets = [
            random.uniform(0.0, 2.0 * math.pi)
            for _ in range(MOVING_BOX_COUNT)
        ]
        self.moving_box_phase = 0.0

    def sample_position_away_from_paths(
            self,
            object1,
            object2,
            min_distance=MIN_SPAWN_DISTANCE,
            max_attempts=5000):
        for _ in range(max_attempts):
            x = random.uniform(-SPAWN_LIMIT, SPAWN_LIMIT)
            y = random.uniform(-SPAWN_LIMIT, SPAWN_LIMIT)
            if not is_place(
                    x,
                    y,
                    object1=object1,
                    object2=object2,
                    min_distance=min_distance):
                continue
            if all(
                self.point_to_path_distance(x, y, path) >= min_distance
                for path in self.moving_box_paths
            ):
                return x, y
        raise RuntimeError(
            "Failed to sample a position clear of moving obstacle paths."
        )

    def update_moving_obstacles(self, advance=True):
        if advance:
            self.moving_box_phase += MOVING_BOX_PHASE_STEP

        moving_positions = []
        for path, phase_offset in zip(
                self.moving_box_paths,
                self.moving_box_phase_offsets):
            axis, center_x, center_y = path
            displacement = MOVING_BOX_AMPLITUDE * math.sin(
                self.moving_box_phase + phase_offset
            )
            if axis == "x":
                moving_positions.append((center_x + displacement, center_y))
            else:
                moving_positions.append((center_x, center_y + displacement))

        for box_index, (x, y) in enumerate(moving_positions):
            self.box_translations[box_index].setSFVec3f([x, y, 0.05])
            self.box[box_index].setVelocity([0, 0, 0, 0, 0, 0])

        self.object1 = [list(position) for position in moving_positions]

    def relocate_object(self, object_index):
        object_positions = list(self.object2)
        object_positions[object_index] = None
        robot_pos = self.trans.getSFVec3f()
        blockers = self.object1 + [[robot_pos[0], robot_pos[1]]]
        x, y = self.sample_position_away_from_paths(
            object1=blockers,
            object2=object_positions,
            min_distance=BALL_SPAWN_DISTANCE
        )

        self.object2[object_index] = [x, y]
        wz = self.objects[object_index].getField('translation')
        wz.setSFVec3f([x, y, 0.04])
        self.objects[object_index].setVelocity([0, 0, 0, 0, 0, 0])
        self.objects[object_index].resetPhysics()

    def get_state(self, action=np.zeros(2)):
        img = self.camera.getImageArray()
        img = np.array(img)
        img = np.average(
            img.reshape((84, 84, 3)),
            axis=-1,
        ).reshape((1, 84, 84)) / 255.0

        pos = np.array(self.trans.getSFVec3f())
        rot = np.array(self.rotation.getSFRotation())
        pos_rot = np.zeros(5)
        pos_rot[0] = pos[0]
        pos_rot[1] = pos[1]
        a, b, pos_rot[2] = self.quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])
        pos_rot[3] = (pos[0]-self.x)*50
        pos_rot[4] = (pos[1]-self.y)*50
        self.x=pos[0]
        self.y=pos[1]
        state = [
            img,
            pos_rot.reshape((1, 5))
        ]
        return state

    def quaternion_to_euler(self, qx, qy, qz, qw, degree_mode=0):

        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))

        pitch = 0
        sinp = 2 * (qw * qy - qz * qx)
        if (abs(sinp) >= 1):
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        # degree_mode=1:銆愯緭鍑恒€戞槸瑙掑害鍒讹紝鍚﹀垯寮у害鍒?
        if degree_mode == 1:
            roll = np.rad2deg(roll)
            pitch = np.rad2deg(pitch)
            yaw = np.rad2deg(yaw)
        euler = np.array([roll, pitch, yaw])

        return euler

    # 娆ф媺瑙掕浆鍥涘厓鏁?
    # ================OKOK
    def euler_to_quaternion(self, euler, degree_mode=1):
        roll, pitch, yaw = euler
        # degree_mode=1:銆愯緭鍏ャ€戞槸瑙掑害鍒讹紝鍚﹀垯寮у害鍒?
        if degree_mode == 1:
            roll = np.deg2rad(roll)
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)

        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        q = np.array([qw, qx, qy, qz])
        return q

    def reset(self):

        # 閲嶆柊璁剧疆鍒濆淇℃伅
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        self.left_wheel_motor.setPosition(float('inf'))
        self.right_wheel_motor.setPosition(float('inf'))

        self.now_step = 0
        self.sample_instruction()
        # 璁剧珛灏忚溅鍒濆浣嶇疆
        self.set_car_target()

        self.robot.step(self.timestep)
        return self.get_state()

    def set_car_target(self):
        #鏀剧疆鍥涗釜鐩掑瓙
        self.object1=[]
        self.object2=[]

        self.sample_moving_obstacle_layout()
        self.update_moving_obstacles(advance=False)
        for box in self.box:
            box.resetPhysics()

        # Spawn the robot after obstacles so it cannot overlap any box.
        x, y = self.sample_position_away_from_paths(
            object1=self.object1,
            object2=self.object2,
        )
        self.x=x
        self.y=y
        self.trans.setSFVec3f([x, y, 0])
        self.name.resetPhysics()
        robot_blocker = [[x, y]]

        #鏀剧疆灏忕悆
        for i in range(self.object_num):
            x, y = self.sample_position_away_from_paths(
                object1=self.object1 + robot_blocker,
                object2=self.object2,
                min_distance=BALL_SPAWN_DISTANCE
            )
            self.object2.append([x,y])
            wz=self.objects[i].getField('translation')
            wz.setSFVec3f([x,y,0.04])
            self.objects[i].setVelocity([0, 0, 0, 0, 0, 0])
            self.objects[i].resetPhysics()
        '''
        x = 1.2
        y = 1.1
        self.target[0] = -1
        self.target[1] = -1
        self.trans.setSFVec3f([x, y, 0])
        r = (x +1) ** 2 + (y+1) ** 2
        self.last_distance = pow(r, 0.5)
        '''

    def delay(self, n):
        for i in range(n):
            for j in range(self.object_num):
                self.objects[j].setVelocity([0, 0, 0, 0, 0, 0])
            self.update_moving_obstacles()
            self.robot.step(self.timestep)

    def step(self, action):
        self.now_step += 1

        speed_limit = np.nextafter(float(self.speed), 0.0)
        left_speed = np.clip(
            (float(action[0]) + 1.0) * self.speed / 2.0,
            0.0,
            speed_limit,
        )
        right_speed = np.clip(
            (float(action[1]) + 1.0) * self.speed / 2.0,
            0.0,
            speed_limit,
        )
        self.left_wheel_motor.setVelocity(left_speed)
        self.right_wheel_motor.setVelocity(right_speed)

        self.delay(self.delay_time)

        state = self.get_state(action)
        done = False
        target = False
        reward = 0.1 * action[0] * action[1] - 0.1
        pos = state[1][0]

        for i in range(self.object_num):
            this_distance = math.pow(((pos[0] - self.object2[i][0]) ** 2 + (pos[1] - self.object2[i][1]) ** 2), 0.5)
            if this_distance < 0.1:
                reward += 10.0
                self.relocate_object(i)
                break

        if wall_detection(pos[0], pos[1]) == True:
            reward -= 10.0
            target = True
        elif box_detection(pos[0], pos[1], self.object1) == True:
            reward -= 10.0
            target = True

        if self.now_step >= self.max_step:
            done = True
        next_state = self.get_state(action)
        return next_state, reward, done, target

if __name__ == '__main__':
    env = env(1, 100)
    plt.ion()
    for e in range(100):
        rw=0
        env.reset()
        print(e)

        for i in range(10000):
            action1= 0.5
            action2 = 0
            state, reward, done,target = env.step([action1, action2])

            plt.imshow(state[0].transpose(1, 2, 0))
            plt.pause(0.0001)
            plt.clf()
            print(i,reward)
            if(target or done):
                break
            rw+=reward
        print(rw)

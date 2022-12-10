import os
import cv2
import pickle
import random
import numpy as np
import math


class SortOfClevr:
    def __init__(self, train_size=9800, test_size=200, image_size=75, object_size=5, question_dim=11, num_of_question=10, save_dir='./data'):
        self.train_size = train_size
        self.test_size = test_size
        self.image_size = image_size
        self.object_size = object_size
        self.question_dim = question_dim # 6 for one-hot vector of color, 2 for question type, 3 for question subtype
        self.num_of_question = num_of_question # Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]
        self.save_dir = save_dir # directory to store training data
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (0,156,255), (128,128,128), (0,255,255)] #r,g,b,o,k,y
        self.nonrel_index = len(self.colors)
        self.rel_index = len(self.colors) + 1
        self.subtype_index_start = len(self.colors) + 2

        self.fix_random_seed()

        self.train_datasets = []
        self.test_datasets = []
        self.train_objects = []
        self.test_objects = []

    def fix_random_seed(self):
        random.seed(1)
        np.random.seed(1)

    def make_directory(self):
        try:
            os.makedirs(self.save_dir)
        except:
            print('Data directory already exists.')

    def generate_new_center(self, objects):
        # if an object is already present, then find center that is 2X away for all old centers
        # if no old object is present, then no need to check anything
        while True:
            pas = True
            center = np.random.randint(self.object_size, self.image_size - self.object_size, 2)        
            if len(objects) > 0:
                for _, c, _ in objects:
                    if ((center - c) ** 2).sum() < ((self.object_size * 2) ** 2):
                        pas = False
            if pas:
                return center

    def generate_one_image(self):
        # function create 1 image and list of location,color,shape of 6 objects
        objects = []
        # create an image with white background
        image = np.ones((self.image_size, self.image_size, 3)) * 255
        # Each image should have six objects, one for each color in the color list
        for color_id, color in enumerate(self.colors):
            center = self.generate_new_center(objects)
            # take 50% of objects as circles and 50% as rectangles
            if random.random() < 0.5:
                upper_left = (center[0] - self.object_size, center[1] - self.object_size) # xmin,ymin
                lower_right = (center[0] + self.object_size, center[1] + self.object_size) # xmax,ymax
                cv2.rectangle(image, upper_left, lower_right, color, -1)
                objects.append((color_id, center, 'rectangle'))
            else:
                circle_center = (center[0], center[1])
                cv2.circle(image, circle_center, self.object_size, color, -1)
                objects.append((color_id, center, 'circle'))
        image = image / 255. # normalize image
        return objects, image

    def generate_nonrelational_questions(self, objects):
        norel_questions, norel_answers = [], []
        for idx in range(self.num_of_question):
            question = np.zeros(self.question_dim)
            color = random.randint(0, len(self.colors) - 1)
            question[color] = 1
            question[self.nonrel_index] = 1
            subtype = random.randint(0,2)
            question[subtype + self.subtype_index_start] = 1
            norel_questions.append(question)
            if subtype == 0:
                """query shape->rectangle/circle"""
                if objects[color][2] == 'rectangle':
                    answer = 2
                else:
                    answer = 3

            elif subtype == 1:
                """query horizontal position->yes/no"""
                if objects[color][1][0] < self.image_size / 2:
                    answer = 0
                else:
                    answer = 1

            elif subtype == 2:
                """query vertical position->yes/no"""
                if objects[color][1][1] < self.image_size / 2:
                    answer = 0
                else:
                    answer = 1
            norel_answers.append(answer)
        return norel_questions, norel_answers

    def generate_relational_questions(self, objects):
        rel_questions, rel_answers = [], []
        for idx in range(self.num_of_question):
            question = np.zeros((self.question_dim))
            color = random.randint(0, len(self.colors) - 1)
            question[color] = 1
            question[self.rel_index] = 1
            subtype = random.randint(0,2)
            question[subtype + self.subtype_index_start] = 1
            rel_questions.append(question)

            if subtype == 0:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = math.inf
                closest = dist_list.index(min(dist_list))
                if objects[closest][2] == 'rectangle':
                    answer = 2
                else:
                    answer = 3
                    
            elif subtype == 1:
                """furthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                furthest = dist_list.index(max(dist_list))
                if objects[furthest][2] == 'rectangle':
                    answer = 2
                else:
                    answer = 3

            elif subtype == 2:
                """count->1~6"""
                my_obj = objects[color][2]
                count = -1
                for obj in objects:
                    if obj[2] == my_obj:
                        count +=1 
                answer = count+4

            rel_answers.append(answer)
        return rel_questions, rel_answers

    def generate_questions_per_image(self, objects):
        norel_questions, norel_answers = self.generate_nonrelational_questions(objects)
        rel_questions, rel_answers = self.generate_relational_questions(objects)
        relational = (rel_questions, rel_answers)
        nonrelational = (norel_questions, norel_answers)
        return relational, nonrelational

    def parse_object_info(self, objects, index):
        save_objects = []
        for one_object in objects:
            color_id, center, shape = one_object
            if shape == 'rectangle':
                shape_id = 0
            elif shape == 'circle':
                shape_id = 1
            save_objects.append([str(index).zfill(4), color_id, center[0], center[1], shape_id, self.object_size])
        return save_objects

    def creat_dataset(self):
        for train_index in range(self.train_size):
            objects, image = self.generate_one_image()
            relational, nonrelational = self.generate_questions_per_image(objects)
            self.train_datasets.append((image, relational, nonrelational))
            self.train_objects.append(self.parse_object_info(objects, train_index))

        for test_index in range(self.test_size):
            objects, image = self.generate_one_image()
            relational, nonrelational = self.generate_questions_per_image(objects)
            self.test_datasets.append((image, relational, nonrelational))
            self.test_objects.append(self.parse_object_info(objects, test_index))

    def save_dataset(self):
        self.make_directory()
        pixels_filename_train = os.path.join(self.save_dir,'sort-of-clevr-pixels-train.pickle')
        pixels_filename_test = os.path.join(self.save_dir,'sort-of-clevr-pixels-test.pickle')
        states_filename_train = os.path.join(self.save_dir,'sort-of-clevr-states-train.pickle')
        states_filename_test = os.path.join(self.save_dir, 'sort-of-clevr-states-test.pickle')

        with open(pixels_filename_train, 'wb') as f:
            pickle.dump(self.train_datasets, f)

        with open(pixels_filename_test, 'wb') as f:
            pickle.dump(self.test_datasets, f)

        with open(states_filename_train, 'wb') as f:
            pickle.dump(self.train_objects, f)

        with open(states_filename_test, 'wb') as f:
            pickle.dump(self.test_objects, f)


if __name__ == '__main__':
    sort_of_clevr = SortOfClevr()
    sort_of_clevr.creat_dataset()
    sort_of_clevr.save_dataset()

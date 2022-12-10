import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import torch


def visualize_image(image_mat):
    return np.dstack((image_mat[:,:,2], image_mat[:,:,1], image_mat[:,:,0]))

def decode_questions(question):
    colors = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
    color = colors[question.tolist()[0:6].index(1)] 
    if question[6] == 1:
        if question[8] == 1:
            query = 'Q: What is the shape of the {} object?'.format(color)
        if question[9] == 1:
            query = 'Q: Is the {} object on the left of the image?'.format(color)
        if question[10] == 1:
            query = 'Q: Is the {} object on the top of the image?'.format(color)
    if question[7] == 1:
        if question[8] == 1:
            query = 'Q: What is the shape of the object closest to the {} one?'.format(color)
        if question[9] == 1:
            query = 'Q: What is the shape of the object furthest from the {} one?'.format(color)
        if question[10] == 1:
            query = 'Q: How many objects have the same shape as {} object?'.format(color)
    return query    

def decode_answers(answer):
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']
    return "A: {}".format(answer_sheet[answer])

def show_one_question(image, question, answer):
    fig, axes = plt.subplots(1,2, figsize=(7, 2.5), gridspec_kw={'width_ratios': [1, 2]})
    axes[0].set_aspect('equal')
    axes[0].imshow(visualize_image(image))
    axes[0].set_yticks([])
    axes[0].set_xticks([])

    question_string, answer_string = decode_questions(question), decode_answers(answer)
    axes[1].text(0.025, 0.6, question_string, fontsize=11)
    axes[1].text(0.025, 0.4, answer_string, fontsize=11)

    axes[1].axis('off')
    plt.tight_layout()


def get_data_loader(data_file, shuffle=True, batch_size=64):
    images_list = []
    question_list = []
    answer_list = []
    indicator_list = []

    with open(data_file, 'rb') as f:
        image_questions_pair = pickle.load(f)
        
    for image, relations, nonrelations in image_questions_pair[:5000]:
        image = np.swapaxes(image, 0, 2)

        for question, answer in zip(relations[0], relations[1]):
            images_list.append(image)
            question_list.append(question)
            answer_list.append(answer)
            indicator_list.append(1)
            
        for question, answer in zip(nonrelations[0], nonrelations[1]):
            images_list.append(image)
            question_list.append(question)
            answer_list.append(answer)
            indicator_list.append(0)
        
    images = torch.FloatTensor(np.asarray(images_list))
    questions = torch.FloatTensor(np.array(question_list))
    answers = torch.LongTensor(np.asarray(answer_list).reshape(-1,1))
    indicators = torch.LongTensor(np.asarray(indicator_list).reshape(-1,1))

    data = torch.utils.data.TensorDataset(images, questions, answers, indicators)
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader
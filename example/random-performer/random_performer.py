import random
import json


def main():
    with open('/config/config.json', 'r') as fp:
        config = json.load(fp)

    with open('/input/input.csv', 'r') as data_file:
        data = list(map(lambda line: line.split(','), data_file.read().splitlines()))

    if config['Task'] == '2a':
        if config['Mode'] == 'Train':
            train(data, config['DataDirectory'])
        elif config['Mode'] == 'OnlineTrain':
            online_train(data, config['DataDirectory'])
        elif config['Mode'] == 'Test':
            task_2a(data, config['DataDirectory'])

    elif config['Task'] == '2b':
        if config['Mode'] == 'Test':
            task_2b(data, config['DataDirectory'])


def train(data, data_directory):
    # gather all the possible class labels
    class_labels = set()
    for image_filename, class_label in data:
        class_label = int(class_label)
        class_labels.add(class_label)

    # save the labels
    with open('/output/params/labels.csv', 'w') as fp:
        fp.write(','.join(map(str, class_labels)))


def online_train(data, data_directory):
    # read the class labels we already have saved
    class_labels = set()
    with open('/params/labels.csv', 'r') as fp:
        class_labels.update(fp.read().split(','))

    # go through new training data and add in any new class labels
    for image_filename, class_label in data:
        class_label = int(class_label)
        class_labels.add(class_label)

    # save the labels
    with open('/output/params/labels.csv', 'w') as fp:
        fp.write(','.join(map(str, class_labels)))


def task_2a(data, data_directory):
    # read in class labels
    with open('/params/labels.csv', 'r') as fp:
        class_labels = list(map(int, fp.readlines()[0].split(',')))

    # calculate scores for each class label
    results = []
    for trial_id, image_filename in data:
        results.append((trial_id, [random.uniform(0, 1) for _ in class_labels]))

    # output results
    with open('/output/output.csv', 'w') as output_file:
        # write the header row that indicates which score pertains to which class label
        output_file.write('trial-id,{}\n'.format(','.join(map(str, class_labels))))

        # write the actual results for each trial
        for trial_id, scores in results:
            output_file.write('{},{}\n'.format(trial_id, ','.join(map(str, scores))))


def task_2b(data, data_directory):
    # calculate scores for each candidate
    results = []
    for trial_id, probe_filename, *candidate_filenames in data:
        results.append((trial_id, [random.uniform(0, 1) for _ in candidate_filenames]))

    # output results
    with open('/output/output.csv', 'w') as output_file:
        # write the actual results for each trial
        for trial_id, scores in results:
            output_file.write('{},{}\n'.format(trial_id, ','.join(map(str, scores))))


if __name__ == '__main__':
    main()

import os
import os.path as osp
import itertools
import subprocess
import random,string
import datetime
import re
import pandas as pd
from tqdm import tqdm

os.environ['PYTHONWARNINGS']="ignore"

class Runner:

    def __init__(self, base_command):
        self.res = {'seed': [], 'test_acc': []}
        self.base_command = base_command
        # self.base_file = 'runner.out'
        timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")
        if not osp.exists('output_runner/'):
           os.mkdir('output_runner')
        self.base_file = "output_runner/runner" + "-" + timstr + "-" + self._random_str() + ".out"

    def multi_run(self, seeds):
        for seed in tqdm(seeds):
            self.res['seed'].append(seed)
            command = self.base_command
            command = command.replace('--seed xxx', '--seed '+f'{seed}')
            self.run(command)
            self.write_res()
        self.get_df()

    def run(self, command):
        with open(self.base_file, "w") as file:
            subprocess.run(command.split(), stdout=file)

    def write_res(self):
        with open(self.base_file, "r") as file:
            for line in file.readlines():
                if 'Test set' in line:
                    test_acc = float(re.findall("\d+\.\d+", line)[-1])

        # print(test_acc)
        self.res['test_acc'].append(test_acc)

    def get_df(self):
        df = pd.DataFrame.from_dict(self.res)
        df = df.sort_values(by=['seed'], ascending=True)
        print(df)
        print('%s test_acc: %s +- %s' % (dataset, df.test_acc.mean(), df.test_acc.std()))
        self.df = df

    def save(self):
        self.df.to_csv(f'runner.csv', index=False)

    def _random_str(self, randomlength=3):
        a = list(string.ascii_letters)
        random.shuffle(a)
        return ''.join(a[:randomlength])


seeds = list(range(10, 20))

scripts = ['scripts/assortative/', 'scripts/disassortative/', 'scripts/adversarial']
for path in scripts:
    print('=== ' + path + ' ===')
    for file in os.listdir(path):
        command = []
        with open(osp.join(path, file), 'r') as f:
            for line in f.readlines()[1:]:
                line = line.replace('\\', '').strip()
                if 'seed' in line:
                    line = '--seed xxx'
                command.append(line)
                if 'dataset' in line:
                    dataset = line.split()[-1]
                    print(line)
        command = ' '.join(command)
        runner = Runner(command)
        runner.multi_run(seeds)
    # delete saved/cora* and saved/citeseer*
    # sicne in adversarial robustness experiments, we only use the largest component of the graphs
    for file in os.listdir('saved/'):
        if 'cora' in file or 'citeseer' in file:
            os.remove(osp.join('saved/', file))



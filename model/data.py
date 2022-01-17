import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    # words = []
    # for sent in nltk.sent_tokenize(tokens):
    #     words.append(nltk.word_tokenize(sent))
    # return words
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        path = 'data/squad'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'
        # 数据预处理
        print("preprocessing data files...")
        if not os.path.exists('{}/{}l'.format(path, args.train_file)):
            self.preprocess_file('{}/{}'.format(path, args.train_file))
        if not os.path.exists('{}/{}l'.format(path, args.dev_file)):
            self.preprocess_file('{}/{}'.format(path, args.dev_file))

        # 定义数据格式
        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        # include_lengths--是返回填充的minibatch的元组和包含每个示例的长度的列表，还是返回填充的minibatch
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]
        # 如果不是第一次加载数据，执行下面的代码，直接加载数据
        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        # 如果是第一次加载数据，执行
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train='{}l'.format(args.train_file),
                validation='{}l'.format(args.dev_file),
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        # cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[args.train_batch_size, args.dev_batch_size],
                                       device=device,
                                       sort_key=lambda x: len(x.c_word))

    def preprocess_file(self, path):
        dump = []
        abnormals = [' ', '\n']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']     # 取到value，是一个list

            for article in data:    # 对于每一个dict
                for paragraph in article['paragraphs']:     # 对于paragraphs里的每一个dict
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:             # 对于段落的每一个问题
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:           # 对u每一个答案
                            answer = ans['text']
                            # 获取开始答案的位置（字符级别）
                            s_idx = ans['answer_start']
                            # 获取结束答案的位置（字符级别）
                            e_idx = s_idx + len(answer)
                            # 将s_idx和e_idx转为词级别的计数
                            l = 0
                            # 一旦找到了开始的那个词的位置，就可以把下面的flag设置为True
                            flag = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:     # 碰上空格，换行符等需要考虑其长度
                                        l += 1
                                    else:
                                        break
                                # # exceptional cases
                                # if t[0] == '"' and context[l:l + 2] == '\'\'':
                                #     t = '\'\'' + t[1:]
                                # elif t == '"' and context[l:l + 2] == '\'\'':
                                #     t = '\'\''

                                l += len(t)
                                if l > s_idx and flag == False:
                                    s_idx = i
                                    flag = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open('{}l'.format(path), 'w', encoding='utf-8') as f:
        # with open('{}'.format(path), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)

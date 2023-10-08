import os
import sys
import json
import argparse

sys.path.insert(0, 'src/')

from typing import List
from datasets import Dataset, concatenate_datasets

from utils import save_dataset
from tasks import task_map, BaseTask
from logger_config import logger

parser = argparse.ArgumentParser(description='data preprocessing for all tasks')
parser.add_argument('--output-dir', default='./data/tasks/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--template-idx', default=0, type=int, metavar='N',
                    help='template index for the task')
parser.add_argument('--max-train-examples', default=30_000, type=int, metavar='N',
                    help='maximum number of training examples per task')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))


def format_and_save_corpus():
    corpus_list: List[Dataset] = []
    for task_name, task_cls in task_map.cls_dic.items():
        task: BaseTask = task_cls(template_idx=args.template_idx)
        logger.info(f'Task: {task_name}')
        task_corpus: Dataset = task.get_corpus()
        if task_corpus is None:
            continue

        logger.info(f'Task: {task_name}, corpus size: {len(task_corpus)}')
        corpus_list.append(task_corpus)

    corpus: Dataset = concatenate_datasets(corpus_list)
    corpus = corpus.add_column('id', [str(i) for i in range(len(corpus))])

    out_path: str = f'{args.output_dir}/passages.jsonl.gz'
    save_dataset(corpus, out_path=out_path)
    logger.info(f'Save {len(corpus)} lines to {out_path}')


def prepare_split(split: str = 'test'):
    dataset_list: List[Dataset] = []
    for task_name, task_cls in task_map.cls_dic.items():
        task: BaseTask = task_cls(template_idx=args.template_idx)
        logger.info(f'Task: {task_name}')
        task_ds: Dataset = task.get_task_data(split=split)
        if task_ds is None:
            continue

        logger.info(f'Task: {task_name}, size: {len(task_ds)}')
        if split == 'train' and len(task_ds) > args.max_train_examples:
            task_ds = task_ds.shuffle().select(range(args.max_train_examples))
            logger.info(f'Random sample to {len(task_ds)} examples')
        dataset_list.append(task_ds)

    dataset: Dataset = concatenate_datasets(dataset_list)

    out_path: str = os.path.join(args.output_dir, f'{split}.jsonl.gz')
    save_dataset(dataset, out_path)
    logger.info(f'Save {len(dataset)} examples to {out_path}')


def main():
    format_and_save_corpus()
    for split in ['train', 'test']:
        prepare_split(split)


if __name__ == '__main__':
    main()

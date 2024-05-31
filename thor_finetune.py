import argparse
import os
from os.path import join
from addict import Dict

import yaml
import torch
from transformers import GenerationConfig

from src.ft.cot_default import ChainOfThoughtDefault
from src.ft.engine_prompt import PromptTrainer
from src.ft.engine_thor import ThorTrainer
from src.ft.service import RuSentNE2023CodalabService, CsvService
from src.ft.utils import set_seed, load_params_LLM, OutputHandler
from src.ft.loader import MyDataLoader
from src.ft.model import LLMBackbone
from utils import LABEL_MAP_REVERSE, DATA_DIR


class Template:

    def __init__(self, args):
        config = Dict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        if config.eval_iter >= 0:
            config.shuffle = False
        self.config = config

        # Setup COT mode.
        cot_choices = {
            "default": ChainOfThoughtDefault(),
        }
        self.thor_cot = cot_choices[self.config.cot_mode]

        if self.config.instruct is None and self.config.reasoning == "prompt":
            presets = {
                "default": "What's the attitude of the sentence '{context}', to the target '{target}'?",
                "v2": "What is the attitude of the author or another subject in the sentence '{context}' "
                      "to the target '{target}'?",
            }
            self.config.instruct = presets[self.config.cot_mode]

    def forward(self):
        print(f"Loading data. Shuffle mode: {self.config.shuffle}")

        (self.trainLoader, self.validLoader, self.testLoader), self.config = \
            MyDataLoader(config=self.config, thor_cot=self.thor_cot).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)
        print("Learning Rate (for training): ", self.config.bert_lr)
        print("Model Temperature: ", self.config.temperature)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            print("Prompt: {}".format(self.config.instruct))
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print(f"Choosing THoR multi-step infer mode. [{type(self.thor_cot.__class__.__name__)}]")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader,
                                  cot=self.thor_cot)
        else:
            raise Exception('Should choose a correct reasoning mode: prompt or thor.')

        e_load = None
        epoch_from = 0
        do_zero_shot = self.config.zero_shot == True
        do_final_evaluation = self.config.eval_iter >= 0
        do_train = not do_zero_shot and not do_final_evaluation
        if self.config.load_iter >= 0:
            e_load = self.config.load_iter if self.config.load_iter >= 0 else None
            print(f"Loading the pre-trained state: {e_load}")
            trainer.load_from_epoch(epoch=self.config.load_iter)
            epoch_from = e_load + 1
            if do_train:
                # We need to make sure that the epochs are correct in the case when we continue training process.
                assert (self.config.load_iter < self.config.epoch_size)
                # Register the result so we know the best state before.
                r = trainer.evaluate_step(self.validLoader, 'valid')
                trainer.add_instance(r)
        if do_zero_shot:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return
        if do_final_evaluation:
            print(f"Final evaluation. Loading state: {self.config.eval_iter}")
            h = OutputHandler()
            if self.config.reasoning == 'thor':
                trainer.output_handler = lambda text: h.forward(text)
            r = trainer.final_evaluate(self.config.eval_iter)
            print(r)
            submission_name = f"{self.config.model_path.replace('/', '_')}-{self.config.eval_iter}-test-submission.zip"
            RuSentNE2023CodalabService.save_submission(target=join(self.config.preprocessed_dir, submission_name),
                                                       labels=[LABEL_MAP_REVERSE[l] for l in trainer.preds['total']])

            CsvService.write(lines_it=h.iter_chunks(3),
                             target=join(self.config.preprocessed_dir, submission_name + '.gen.csv'),
                             header=["s1_aspect", "s2_opinion", "s3_polarity"])
            return

        print("Fine-tuning mode for training.")
        trainer.train(epoch_from=epoch_from)


if __name__ == '__main__':

    gen_config = GenerationConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-es', '--epoch_size', default=1, type=int)
    parser.add_argument('-e', '--eval_iter', default=-1, type=int, help='running evaluation on specific index')
    parser.add_argument('-d', '--data_name', default='rusentne2023')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-li', '--load_iter', default=-1, type=int, help='load a state on specific index')
    parser.add_argument('-t', '--temperature', default=gen_config.temperature, type=float,
                        help="Necessary for zero-shot option. For the training the default value of the "
                             "configuration from the `transformers` is better since we wish to get the same"
                             "result independing of the chosen path during generation.")
    parser.add_argument('-p', '--instruct', default=None, type=str,
                        help="instructive prompt for `prompt` training engine that involves `context` and `target`"
                             "parameter without need of declaring output labels.")
    parser.add_argument('-bs', '--batch_size', default=None, type=int)
    parser.add_argument('-cm', '--cot_mode', default='default',
                        help="This is a Chain-of-Thought preset name parameter, necessary for "
                             "chosing the chains for the task.")
    parser.add_argument('-bf16', '--use_bf16', action='store_true', default=False,
                        help='Initializing Flan-T5 with torch.bfloat16')

    if not os.path.exists(join(DATA_DIR, "preprocessed")):
        os.makedirs(join(DATA_DIR, "preprocessed"))

    args = parser.parse_args()
    template = Template(args)
    template.forward()

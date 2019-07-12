import math
import multiprocessing
import os
import queue
import statistics
import subprocess
import time
from argparse import ArgumentParser, Namespace

import torch
import yaml
from tqdm import tqdm

import utils
from dataset import load_data
from train import build_model
from utils import misc_utils


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='default.yaml', type=str,
                        help="config file")
    parser.add_argument('--test-src-file', type=str,
                        help="test source file")
    parser.add_argument('--prediction-file', type=str,
                        help="prediction output file")
    parser.add_argument('--mode', default='eval', type=str,
                        help="Mode selection")
    parser.add_argument('--gpu', default=0, type=int,
                        help="Use CUDA on the device.")
    parser.add_argument('--restore', action='store_true',
                        help="restore checkpoint")
    parser.add_argument('--pretrain', type=str,
                        help="load pretrain encoder")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--beam-size', default=1, type=int)
    parser.add_argument('--use-cuda', action='store_true',
                        help="save individual checkpoint")
    parser.add_argument('--model', default='tensor2tensor', type=str)
    return parser.parse_args()



class DescriptionGenerator(object):
    def __init__(self, config, **opt):
        # Load config used for training and merge with testing options
        self.config = yaml.load(open(config, "r"))
        self.config = Namespace(**{**self.config, **opt})

        # Load training data.pkl for src and tgt vocabs
        self.data = load_data(self.config)

        # Load trained model checkpoints
        device, devices_ids = misc_utils.set_cuda(self.config)
        self.model, _ = build_model(None, self.config, device)
        self.model.eval()

    def predict(self, original_src: list) -> list:
        src_vocab = self.data["src_vocab"]
        tgt_vocab = self.data["tgt_vocab"]
        srcIds = src_vocab.convertToIdx(list(original_src), utils.UNK_WORD)
        src = torch.LongTensor(srcIds).unsqueeze(0)
        src_len = torch.LongTensor([len(srcIds)])

        if self.config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        with torch.no_grad():

            if self.config.beam_size > 1:
                # TODO Fix later when using knowledge.
                samples, alignments = self.model.beam_sample(
                    src, src_len, knowledge=self.config.knowledge,
                    knowledge_len=0, beam_size=self.config.beam_size, eval_=False
                )
            else:
                samples, alignments = self.model.sample(src, src_len)

        assert len(samples) == 1
        candidates = [tgt_vocab.convertToLabels(samples[0], utils.EOS)]

        # Replace unk with src tokens
        if self.config.unk and self.config.attention != "None":
            #TODO remove source text in the final output
            s = original_src
            c = candidates[0]
            align = alignments[0]
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
        return cand


class DescriptionGeneratorProxy(object):
    @staticmethod
    def enqueue_output(out, queue):
        for line in iter(out.readline, b""):
            queue.put(line)
        out.close()

    def __init__(self, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.process = subprocess.Popen(
            ["python", "api.py"],
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        self.stdout_reader = multiprocessing.Queue()
        self.stdout_reader_process = multiprocessing.Process(
            target=DescriptionGeneratorProxy.enqueue_output,
            args=(self.process.stdout, self.stdout_reader),
            daemon=True,
        )
        self.stdout_reader_process.start()

    def send(self, src):
        self.process.stdin.write(src + "\n")
        self.process.stdin.flush()

    def recv(self, timeout=None):
        try:
            stdout = self.stdout_reader.get(timeout=timeout)
            return stdout.strip()
        except queue.Empty:
            return ""

    def flush(self):
        while True:
            line = self.recv()
            if line == "COMPLETE":
                break


class DescriptionGeneratorMultiprocessing(object):
    def __init__(self, n_gpus=8, n_process_per_gpu=8, **kwargs):
        self.proxies = []
        for gpu_id in range(n_gpus):
            for _ in range(n_process_per_gpu):
                self.proxies.append(DescriptionGeneratorProxy(gpu_id))
        for proxy in self.proxies:
            proxy.flush()

    def _predict_batch(self, src_list):
        """Batch size = n_gpu * n_process_per_gpu"""
        assert len(src_list) <= len(self.proxies)
        for proxy, src in zip(self.proxies, src_list):
            proxy.send(src)
        return [proxy.recv() for proxy, src in zip(self.proxies, src_list)]

    def predict_all(self, src_list):
        tgt_list = []
        for idx in tqdm(range(int(math.ceil(len(src_list) / len(self.proxies))))):
            tgt_list += self._predict_batch(
                src_list[idx * len(self.proxies) : (idx + 1) * len(self.proxies)]
            )
        return tgt_list


if __name__ == "__main__":
    args = get_args()

    generator = DescriptionGenerator(
        config=args.config,
        gpu=args.gpu,
        restore=args.restore,
        pretrain=args.pretrain,
        mode=args.mode,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        scale=1,
        char=False,
        use_cuda=args.use_cuda,
        seed=1234,
        model=args.model,
    )

    src_file = args.test_src_file
    prediction_file = args.prediction_file
    predictions = []

    pred_times = []

    with open(src_file, 'r') as f:
        for line in f:
            src_text = [line]
            start = time.time()
            pred = generator.predict(src_text)
            end = time.time()
            pred_times.append(end - start)
            prediction = ' '.join().replace('\n', ' ')
            predictions.append(prediction)

    with open(prediction_file, 'w') as f:
        for pred in predictions:
            f.write("%s\n" % pred)

    average_pred_time = statistics.mean(pred_times)
    print("Average Prediction Time Mean:", average_pred_time)

import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

import utils
from wavlm import WavLM, WavLMConfig

def process(filename, in_dir,out_dir,sr):

    save_name=filename.replace(in_dir,out_dir)
    save_name=save_name.replace(".wav", ".pt")
    save_dir= os.path.dirname(save_name)    
    
    os.makedirs(save_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=sr)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav)
    torch.save(c.cpu(), save_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate") 
    parser.add_argument("--in_dir", type=str, default="./dataset/DUMMY_ART", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="./dataset/wavlm", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")
    
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    
    
    for filename in tqdm(filenames):
        process(filename,args.in_dir,args.out_dir,args.sr)
        
    
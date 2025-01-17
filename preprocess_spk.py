import os, sys
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from tqdm import tqdm
from glob import glob 
import argparse


def process(filename, in_dir,out_dir,sr):

    save_name=filename.replace(in_dir,out_dir)
    save_name=save_name.replace(".wav", ".npy")
    save_dir= os.path.dirname(save_name)    
    os.makedirs(save_dir, exist_ok=True)

    fpath = Path(filename)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    np.save(save_name, embed, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate") 
    parser.add_argument("--in_dir", type=str, default="dataset/DUMMY", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="dataset/spk", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading Speaker encoder...")
    encoder = SpeakerEncoder("speaker_encoder/ckpt/pretrained_bak_5805000.pt") 
    print("The speaker encoder has been loaded ! ")
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    print(f"Starting to process {len(filenames)} speaker embeddings.")
    
    
    for filename in tqdm(filenames):
        process(filename,args.in_dir,args.out_dir,args.sr)
        

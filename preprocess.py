import os
import shutil
from data import kss
import hparams as hp

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path
    meta = hp.meta_name
    textgrid_name = hp.textgrid_name
    textgrid_path=hp.textgrid_path

    #makes dirs of preprocessed data
    mel_out_dir = os.path.join(out_dir, "mel")
    os.makedirs(mel_out_dir, exist_ok=True)
    print(f">> {mel_out_dir} exists currently.")

    ali_out_dir = os.path.join(out_dir, "alignment")
    os.makedirs(ali_out_dir, exist_ok=True)
    print(f">> {ali_out_dir} exists currently.")

    f0_out_dir = os.path.join(out_dir, "f0")
    os.makedirs(f0_out_dir, exist_ok=True)
    print(f">> {f0_out_dir} exists currently.")

    energy_out_dir = os.path.join(out_dir, "energy")
    os.makedirs(energy_out_dir, exist_ok=True)
    print(f">> {energy_out_dir} exists currently.")

    #check and unzip TextGrids
    textgrids_dir = os.path.join(out_dir, textgrid_name.replace(".zip", ""))
    
    if os.path.exists(textgrids_dir):
        print(f">> {textgrids_dir} exists currently.") 
    else:
        print(f">> {textgrids_dir} not exists, Unzippng {textgrid_path}{textgrid_name}.") 
        os.system(f'unzip {os.path.join(textgrid_path, textgrid_name)} -d {os.path.join(out_dir,textgrid_name.replace(".zip",""))}')
        print(f'>> Unzipped {textgrid_path}{textgrid_name} successfully.')
        
    train, val = kss.build_from_path(in_dir, out_dir, meta)

    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    main()

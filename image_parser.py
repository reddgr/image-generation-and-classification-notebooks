from PIL import Image
import os
import pandas as pd
import torch
import random
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
import sys
from IPython.display import clear_output
from tqdm import tqdm

class ImageParser:
    def __init__(self, hf_token, read_path, write_path, pkl_dir='pkl'):
        self.hf_token = hf_token
        self.read_path = read_path
        self.write_path = write_path
        self.pkl_dir = pkl_dir
        if os.path.exists(f"{self.pkl_dir}/images_df.pkl"):
            self.images_df = pd.read_pickle(f"{self.pkl_dir}/images_df.pkl")
            print(f"Loaded dataframe | {self.images_df.shape}")
        else:
            self.images_df = pd.DataFrame(columns=['asset_pointer', 'file_name', 'caption'])

    def caption_image(self, image_path, max_new_tokens = 30, captioner_path='Salesforce/blip-image-captioning-base'):
        try:
            im = Image.open(image_path)
            file_name = os.path.basename(image_path)
            print(f'Processing file {file_name} | {im.format, im.size, im.mode}')
        except (IOError, ValueError) as e:
            print(f"Cannot process file: {image_path}. Error: {e}")
            return None
        device = 0 if torch.cuda.is_available() else -1
        captioner = pipeline("image-to-text", model=captioner_path, max_new_tokens=max_new_tokens, device=device)
        itt_output = captioner(im)
        caption = itt_output[0].get('generated_text')
        return caption
    
    def convert_all_images(self, new_format='png'):
        all_files = os.listdir(self.read_path)

        for file_sample in all_files:
            complete_file_name = file_sample.rsplit('.', 1)[0]
            img_pointer = '-'.join(file_sample.split('-')[:2])

            image_orig = f'{self.read_path}/{file_sample}'
            image_png = f'{self.write_path}/{img_pointer}.{new_format}'

            if os.path.exists(image_png):
                print(f"File {image_png} already exists. Skipping.")
                continue

            try:
                im = Image.open(image_orig)
                im_data = [im.format, im.size, im.mode]
            except (IOError, ValueError) as e:
                print(f"Cannot process file: {file_sample}. Error: {e}")
                continue

            im.save(image_png, format=new_format, lossless=True)
            print(f"Orig. file: {complete_file_name} | {im_data} | To upload: {img_pointer}.{new_format}")

            # Add to DataFrame
            if not self.images_df['asset_pointer'].str.contains(f'{img_pointer}').any():
                new_row = pd.DataFrame([{'asset_pointer': f'{img_pointer}', 
                                         'file_name': f'{img_pointer}.{new_format}', 
                                         'caption': ''}])
                self.images_df = pd.concat([self.images_df, new_row], ignore_index=True)
        # Save DataFrame
        self.images_df['img_url'] = self.images_df['file_name'].apply(lambda x: f"https://talkingtochatbots.com/wp-content/uploads/ttcb-images/{x}")
        self.images_df.to_pickle(f"{self.pkl_dir}/images_df.pkl")

    def caption_all_images(self):
        for index, row in tqdm(self.images_df.iterrows(), total=self.images_df.shape[0], desc="Captioning images"):
            img_pointer = row['asset_pointer']
            image_path = f"{self.write_path}/{img_pointer}.png"
            caption = self.caption_image(image_path)
            if caption:
                self.images_df.loc[index, 'caption'] = caption
                print(f"Captioned image {img_pointer} | {caption}")
            else:
                self.images_df.loc[index, 'caption'] = "Talking to Chatbots dataset - image"
            if index % 5 == 0:
                clear_output(wait=True)
        self.images_df.to_pickle(f"{self.pkl_dir}/images_df.pkl")


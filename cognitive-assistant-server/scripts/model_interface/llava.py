import torch
import time
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from tqdm import tqdm
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def main():
    disable_torch_init()
    previous_videos = ""
    # video = 'videollava/serve/examples/sample_demo_1.mp4'
    # video = '/home/ssrinidh/Sruti/Video-LLaVA/videollava/serve/examples/clean_with_me_parts/part1(split-video.com).mp4'
    file2 = open(r"./output.txt", "w") 
    for i in tqdm(range(1,13)):
        video = '/home/ssrinidh/Sruti/Video-LLaVA/videollava/serve/examples/sandwich/part' + str(i) + '(split-video.com).mp4'
        inp = 'Give me a detailed summary of this video segment that is a part of a larger video that is trying to make a sandwich. Give me a detailed description of what you see happening in the video. Can you describe it in only 2 sentences?'
        # if i != 1:
        #     previous_videos = outputs
        #     inp += " For context, here is what happened in the section before this: " + previous_videos
        model_path = 'LanguageBind/Video-LLaVA-7B'
        cache_dir = 'cache_dir'
        device = 'cuda'
        load_4bit, load_8bit = True, False
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
        video_processor = processor['video']
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles

        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        print(f"{roles[1]}: {inp}")
        start = time.time()
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(time.time()-start)
        file2.write(outputs)
        file2.write('\n')
        print(outputs)
    file2.close()
if __name__ == '__main__':
    main()
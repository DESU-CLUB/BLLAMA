from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def init_BLIP(device):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    if device == 'cuda':
        model = Blip2ForConditionalGeneration.from_pretrained(
         "Salesforce/blip2-opt-2.7b", load_in_8bit=True,torch_dtype=torch.float16, device_map = 'auto')
    else:
        print('Using CPU model')
        model = Blip2ForConditionalGeneration.from_pretrained( "Salesforce/blip2-opt-2.7b",device_map={"": device}, torch_dtype=torch.float32,low_cpu_mem_usage=True)

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)
    processor = processor
    return model,processor

def infer_BLIP2(model,processor,image,device):
    outputs=  ''
    prompts = [
        "This is a picture of",
        "Question: What is in the picture? Answer:",
        "Question: Where is this image depicting? Answer:",
        "Question: Who is in this picture? Answer:",
        "Question: What are the things in the picture doing? Answer:",
        "Question: Why do you think they are doing it? Answer:",
        "Question: What emotion does the person or animal in the image feel? Answer:",
        ]
    for prompt in prompts:
        if device == 'cuda':
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        else:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        outputs+= prompt+generated_text+' '
    return outputs

'''
Testing

model,processor = init_BLIP(device)
image = Image.open('/home/spooky/Downloads/IMG20221214012021.jpg')
infer_BLIP2(model,processor,image,device)'''

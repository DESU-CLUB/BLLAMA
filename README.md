## 🦙🌲🤏 BLLAMA: A BLIP2 + ALPACA-LORA pipeline

# Setup
1. Git clone this repository
2. ```pip install -r requirements.txt```

# Training
 This is just a pipeline involving the use of both ALPACA and BLIP-2, without any prior finetuning. You can refer to the details in ALPACA_LORA's repo [here](https://github.com/tloen/alpaca-lora) and the BLIP-2 training details on their GitHub page [here](https://github.com/salesforce/LAVIS/tree/main/projects/blip2). For the pipeline, I have used their model found on HuggingSpace [here](https://huggingface.co/spaces/Salesforce/BLIP2)

# Inference
1. cd to the cloned repo
2. Run ```python3 generate.py```

# Sample of inference
![My Image](test.jpg)


#TODO:
1. Try to reduce VRAM Usage: It hits around 14GB of VRAM on the 7B Weights when combined with BLIP2
2. Add ability for users to customise their prompts to BLIP-2 in Gradio. This can help finetune the context given from BLIP2 to ALPACA, improving accuracy of generated outputs


## Acknowledgements
Once again, I would like to credit the Salesforce team for creating BLIP2, as well as tloen, the original creator of alpaca-lora.

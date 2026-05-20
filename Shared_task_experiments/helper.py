import gc
import torch

def clear_vram():
    # Delete model/tokenizer variables if they exist in global scope
    global llm, tokenizer
    
    if 'model' in globals():
        del model
    if 'tokenizer' in globals():
        del tokenizer
        
    # Force Python garbage collection
    gc.collect()
    
    # Clear the CUDA cache
    torch.cuda.empty_cache()
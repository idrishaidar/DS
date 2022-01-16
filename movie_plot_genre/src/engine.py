from tqdm import tqdm

def train(data_loader, model, optimizer, device, accumulation_steps):
    model.train()

    for batch_ix, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass

def eval():
    pass

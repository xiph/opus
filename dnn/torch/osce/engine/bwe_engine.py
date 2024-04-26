import torch
from tqdm import tqdm
import sys

def train_one_epoch(model, criterion, optimizer, dataloader, device, scheduler, log_interval=10):

    model.to(device)
    model.train()

    running_loss = 0
    previous_running_loss = 0


    with tqdm(dataloader, unit='batch', file=sys.stdout) as tepoch:

        for i, batch in enumerate(tepoch):

            # set gradients to zero
            optimizer.zero_grad()

            # push batch to device
            for key in batch:
                batch[key] = batch[key].to(device)

            target = batch['x_48']
            x16 = batch['x_16']
            x_up = model.upsampler(x16.unsqueeze(1))

            # calculate model output
            output = model(batch['x_16'].unsqueeze(1), batch['features'])

            # calculate loss
            loss = criterion(target, output.squeeze(1), x_up)

            # calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

            # update learning rate
            scheduler.step()

            # sparsification
            if hasattr(model, 'sparsifier'):
                model.sparsifier()

            # update running loss
            running_loss += float(loss.cpu())

            # update status bar
            if i % log_interval == 0:
                tepoch.set_postfix(running_loss=f"{running_loss/(i + 1):8.7f}", current_loss=f"{(running_loss - previous_running_loss)/log_interval:8.7f}")
                previous_running_loss = running_loss


    running_loss /= len(dataloader)

    return running_loss

def evaluate(model, criterion, dataloader, device, log_interval=10):

    model.to(device)
    model.eval()

    running_loss = 0
    previous_running_loss = 0

    with torch.no_grad():
        with tqdm(dataloader, unit='batch', file=sys.stdout) as tepoch:

            for i, batch in enumerate(tepoch):

                # push batch to device
                for key in batch:
                    batch[key] = batch[key].to(device)

                target = batch['x_48']

                # calculate model output
                output = model(batch['x_16'].unsqueeze(1), batch['features'])

                # calculate loss
                loss = criterion(target, output.squeeze(1), model.upsampler(batch['x_16'].unsqueeze(1)))

                # update running loss
                running_loss += float(loss.cpu())

                # update status bar
                if i % log_interval == 0:
                    tepoch.set_postfix(running_loss=f"{running_loss/(i + 1):8.7f}", current_loss=f"{(running_loss - previous_running_loss)/log_interval:8.7f}")
                    previous_running_loss = running_loss


        running_loss /= len(dataloader)

        return running_loss
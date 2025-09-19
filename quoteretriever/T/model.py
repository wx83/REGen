import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import random
from dataset import InterviewDataset, GroupedBatchSampler
from pathlib import Path
import loader
import wandb
from torch.utils.data import ConcatDataset
import datetime
embedder = SentenceTransformer('all-mpnet-base-v2')
embedding_dim = 768  # Dimension for all-MiniLM-L6-v2
from datetime import datetime



def infill_text(article_context, ground_truth):
    """
    Replace the [MASK] token in the article context with the ground truth text.
    """
    return article_context.replace("[MASK]", ground_truth)


import torch
import torch.nn.functional as F
import wandb

def train_bart(train_loader, val_loader, num_epochs, alpha, device, optimizer, bart, embedder, bart_tokenizer, training_script_name, output_dir, batch_sampler, special_token_position, MAX_LENGTH, GRADIENT_ACCUMULATION_STEPS, scheduler, patience):
    for epoch in range(num_epochs):
        bart.train()
        embedder.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_ret_loss = 0.0

        for step, batch in enumerate(train_loader):

            article_context = batch["article_context"] 
            interview_segments = batch["interview_segements"] 
            interview_raw = batch["interview_raw"]
            input_ids = article_context["input_ids"].to(device)
            attention_mask = article_context["attention_mask"].to(device)
            target_ids = interview_segments["input_ids"].to(device)
            target_ids[target_ids == bart_tokenizer.pad_token_id] = -100

            bart_outputs = bart(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids, output_hidden_states=True)
            
            decoder_last_state =  bart_outputs.decoder_hidden_states[-1] 
            last_layer_last_token = decoder_last_state[:, -1, :] 
            generation_loss = bart_outputs.loss 
            retrieval_command_embedding = last_layer_last_token
            retrieval_command_embedding = F.normalize(retrieval_command_embedding, p=2, dim=-1)
            gt_embeddings = embedder.encode(interview_raw, convert_to_tensor=True).to(device) 
            cosine_sim = F.cosine_similarity(retrieval_command_embedding.unsqueeze(1), 
                                             gt_embeddings.unsqueeze(0), dim=-1)
            temperature = 0.1
            cosine_sim = cosine_sim / temperature
            labels_tensor = torch.arange(retrieval_command_embedding.size(0)).to(device)
            retrieval_loss = F.cross_entropy(cosine_sim, labels_tensor)

            loss = generation_loss + alpha * retrieval_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_gen_loss += generation_loss.item()
            total_ret_loss += retrieval_loss.item()
        avg_loss = total_loss / len(train_loader)
        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_ret_loss = total_ret_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Gen Loss: {avg_gen_loss:.4f}, Ret Loss: {avg_ret_loss:.4f}")
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"learning_rate": current_lr})

        date_str = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        checkpoint_dir = output_dir / f"bart_{alpha}_{special_token_position}" / date_str
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        checkpoint = {
            "epoch": epoch + 1,  # next epoch to run
            "bart_state_dict": bart.state_dict(),
            "embedder_state_dict": embedder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "alpha": alpha,
                "special_token_position": special_token_position,
                "max_length": MAX_LENGTH,
                "batch_size": train_loader.batch_size,
                "num_training_files": len(train_loader.dataset),
                "num_validation_files": len(val_loader.dataset),
                # Add any other hyperparameters or config details you need
            }
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        bart.eval()
        embedder.eval()
        total_val_loss = 0.0
        total_val_gen_loss = 0.0
        total_val_ret_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                article_context = batch["article_context"] # i think one mask is missing
                interview_segments = batch["interview_segements"] 
                input_ids = article_context["input_ids"].to(device)
                interview_raw = batch["interview_raw"]
                attention_mask = article_context["attention_mask"].to(device)
                target_ids = interview_segments["input_ids"].to(device)
                target_ids[target_ids == bart_tokenizer.pad_token_id] = -100

                bart_outputs = bart(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids, output_hidden_states=True)
                decoder_last_state =  bart_outputs.decoder_hidden_states[-1] 
                last_layer_last_token = decoder_last_state[:,-1,:]
                generation_loss = bart_outputs.loss 

                retrieval_command_embedding = last_layer_last_token 
                retrieval_command_embedding = F.normalize(retrieval_command_embedding, p=2, dim=-1)
                gt_embeddings = embedder.encode(interview_raw, convert_to_tensor=True).to(device)
                cosine_sim = F.cosine_similarity(retrieval_command_embedding.unsqueeze(1),
                                                    gt_embeddings.unsqueeze(0), dim=-1)
                temperature = 0.1
                cosine_sim = cosine_sim / temperature
                labels_tensor = torch.arange(retrieval_command_embedding.size(0)).to(device)
                retrieval_loss = F.cross_entropy(cosine_sim, labels_tensor)
                total_val_gen_loss += generation_loss.item()
                total_val_ret_loss += retrieval_loss.item()
                batch_val_loss = generation_loss.item() + alpha * retrieval_loss.item()

                total_val_loss += batch_val_loss
                wandb.log({"val_loss": batch_val_loss, "val_generation_loss": generation_loss, "val_retrieval_loss": retrieval_loss})
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_gen_loss = total_val_gen_loss / len(val_loader)
        avg_val_ret_loss = total_val_ret_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Gen Loss: {avg_val_gen_loss:.4f}, Ret Loss: {avg_val_ret_loss:.4f}")

    

if __name__ == "__main__":
    pass
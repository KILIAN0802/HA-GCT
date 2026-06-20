import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.dataloader import get_dataloaders, get_multivsl_loaders
from models.ha_gct import MultiStreamHA_GCT, EarlyFusionHA_GCT

def parse_args():
    parser = argparse.ArgumentParser(description="HA-GCT Ensemble Inference Pipeline")
    parser.add_argument('--data-dir', type=str, default='data/400VSL/processed/27_direct', help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='vsl400', choices=['vsl400', 'multivsl200'], help='Dataset selection')
    parser.add_argument('--split-method', type=str, default='random', choices=['random', 'signer'], help='MultiVSL200 dataset split method')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-point', type=int, default=27, help='Number of joints')
    parser.add_argument('--in-channels', type=int, default=2, help='Input coordinate channels')
    
    # Model paths and types
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, help='Paths to model checkpoint files')
    parser.add_argument('--model-types', type=str, nargs='+', required=True, help='Types of models: "multistream" or "earlyfusion"')
    parser.add_argument('--d-models', type=int, nargs='+', required=True, help='d_model parameter value for each model')
    
    parser.add_argument('--tta', action='store_true', default=True, help='Use Test-Time Augmentation (TTA)')
    parser.add_argument('--dummy-test', action='store_true', help='Use dummy data for self-testing')
    
    return parser.parse_args()

def get_dummy_loaders(args):
    print("WARNING: Dataset files not found. Using generated dummy data for self-test...")
    num_test = 16
    test_data = np.random.randn(num_test, args.in_channels, 64, args.num_point).astype(np.float32)
    # Determine number of classes from loaded checkpoints or default to 400
    test_labels = np.random.randint(0, 400, num_test)
    test_mask = torch.ones((num_test, 64), dtype=torch.bool)
    test_dataset = TensorDataset(torch.FloatTensor(test_data), test_mask, torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_loader

def perturb_speed(batch, rate):
    B, C, T, V = batch.shape
    x = batch.permute(0, 1, 3, 2).reshape(B, C * V, T)
    new_T = max(2, int(T * rate))
    x_resampled = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
    x_final = F.interpolate(x_resampled, size=T, mode='linear', align_corners=False)
    return x_final.reshape(B, C, V, T).permute(0, 1, 3, 2)

def perturb_speed_mask(mask, rate):
    B, T = mask.shape
    m = mask.float().unsqueeze(1)
    new_T = max(2, int(T * rate))
    m_resampled = F.interpolate(m, size=new_T, mode='nearest')
    m_final = F.interpolate(m_resampled, size=T, mode='nearest')
    return m_final.squeeze(1) > 0.5

def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        num_classes = output.size(1)
        actual_topk = [k for k in topk if k <= num_classes]
        if not actual_topk:
            return [0.0] * len(topk)
        
        maxk = max(actual_topk)
        pred = output.topk(maxk, dim=1, largest=True, sorted=True).indices
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        res = []
        for k in topk:
            if k > num_classes:
                res.append(100.0)  # If topk requested is larger than total classes, accuracy is 100%
            else:
                correct_k = correct[:, :k].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / target.size(0)).item())
        return res

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    assert len(args.model_paths) == len(args.model_types) == len(args.d_models), \
        "The number of --model-paths, --model-types, and --d-models must match!"
        
    # Determine max frames based on dataset
    max_frames = 150 if args.dataset == 'multivsl200' else 64
    
    # Load models
    models = []
    num_classes = 400 if args.dataset == 'vsl400' else 199
    
    for idx, (path, mtype, d_model) in enumerate(zip(args.model_paths, args.model_types, args.d_models)):
        print(f"Loading Model {idx+1} ({mtype}, d_model={d_model}) from {path}...")
        
        if mtype == 'earlyfusion':
            model = EarlyFusionHA_GCT(
                num_joints=args.num_point,
                in_channels=args.in_channels,
                d_model=d_model,
                num_ha_gc_blocks=3,
                num_mhsa_layers=2,
                nhead=4,
                num_classes=num_classes,
                dropout=0.5,
                graph_lambda=0.1,
                max_frames=max_frames
            ).to(device)
        else:
            model = MultiStreamHA_GCT(
                num_joints=args.num_point,
                in_channels=args.in_channels,
                d_model=d_model,
                num_ha_gc_blocks=3,
                num_mhsa_layers=2,
                nhead=4,
                num_classes=num_classes,
                dropout=0.5,
                graph_lambda=0.1,
                max_frames=max_frames
            ).to(device)
            
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        models.append(model)
        print(f"  Loaded successfully!")
        
    # Load dataset
    if not args.dummy_test:
        from utils.preprocessing import SkeletonTransforms
        transform = SkeletonTransforms(
            num_joints=args.num_point,
            max_frames=max_frames,
            verbose=False
        )
        if args.dataset == 'multivsl200':
            _, _, test_loader = get_multivsl_loaders(
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform, split_method=args.split_method
            )
        else:
            _, _, test_loader = get_dataloaders(
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform
            )
    else:
        test_loader = get_dummy_loaders(args)
        
    # Run evaluation
    all_ensemble_probs = []
    all_targets = []
    
    use_amp = (device.type == 'cuda')
    
    print("\nRunning ensemble inference...")
    with torch.no_grad():
        for batch_data, batch_mask, batch_labels in tqdm(test_loader, desc="[Ensemble Inference]"):
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)
            
            # Store probabilities for this batch from each model
            batch_probs = []
            
            for model in models:
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        if args.tta:
                            out_orig = model(batch_data, mask=batch_mask)
                            
                            batch_flipped = batch_data.clone()
                            batch_flipped[:, 0, :, :] = -batch_flipped[:, 0, :, :]
                            out_flipped = model(batch_flipped, mask=batch_mask)
                            
                            batch_speed_09 = perturb_speed(batch_data, 0.9)
                            mask_speed_09 = perturb_speed_mask(batch_mask, 0.9)
                            out_speed_09 = model(batch_speed_09, mask=mask_speed_09)
                            
                            batch_speed_11 = perturb_speed(batch_data, 1.1)
                            mask_speed_11 = perturb_speed_mask(batch_mask, 1.1)
                            out_speed_11 = model(batch_speed_11, mask=mask_speed_11)
                            
                            prob_orig = F.softmax(out_orig, dim=-1)
                            prob_flipped = F.softmax(out_flipped, dim=-1)
                            prob_speed_09 = F.softmax(out_speed_09, dim=-1)
                            prob_speed_11 = F.softmax(out_speed_11, dim=-1)
                            
                            model_prob = (prob_orig + prob_flipped + prob_speed_09 + prob_speed_11) / 4.0
                        else:
                            out = model(batch_data, mask=batch_mask)
                            model_prob = F.softmax(out, dim=-1)
                else:
                    if args.tta:
                        out_orig = model(batch_data, mask=batch_mask)
                        
                        batch_flipped = batch_data.clone()
                        batch_flipped[:, 0, :, :] = -batch_flipped[:, 0, :, :]
                        out_flipped = model(batch_flipped, mask=batch_mask)
                        
                        batch_speed_09 = perturb_speed(batch_data, 0.9)
                        mask_speed_09 = perturb_speed_mask(batch_mask, 0.9)
                        out_speed_09 = model(batch_speed_09, mask=mask_speed_09)
                        
                        batch_speed_11 = perturb_speed(batch_data, 1.1)
                        mask_speed_11 = perturb_speed_mask(batch_mask, 1.1)
                        out_speed_11 = model(batch_speed_11, mask=mask_speed_11)
                        
                        prob_orig = F.softmax(out_orig, dim=-1)
                        prob_flipped = F.softmax(out_flipped, dim=-1)
                        prob_speed_09 = F.softmax(out_speed_09, dim=-1)
                        prob_speed_11 = F.softmax(out_speed_11, dim=-1)
                        
                        model_prob = (prob_orig + prob_flipped + prob_speed_09 + prob_speed_11) / 4.0
                    else:
                        out = model(batch_data, mask=batch_mask)
                        model_prob = F.softmax(out, dim=-1)
                        
                batch_probs.append(model_prob)
                
            # Average probabilities across all models: shape (B, num_classes)
            batch_ensemble_prob = torch.stack(batch_probs, dim=0).mean(dim=0)
            
            all_ensemble_probs.append(batch_ensemble_prob.cpu())
            all_targets.append(batch_labels.cpu())
            
    # Concatenate all results
    all_ensemble_probs = torch.cat(all_ensemble_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute Top-1 and Top-5 accuracy
    top1, top5 = topk_accuracy(all_ensemble_probs, all_targets, topk=(1, 5))
    
    print("\n" + "=" * 50)
    print("FINAL ENSEMBLE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total evaluated samples: {len(all_targets)}")
    print(f"TTA Enabled:             {args.tta}")
    print(f"Top-1 Accuracy:          {top1:.2f}%")
    print(f"Top-5 Accuracy:          {top5:.2f}%")
    print("=" * 50)

if __name__ == '__main__':
    main()

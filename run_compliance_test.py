import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os
import numpy as np
import torch
from OAT.DataUtils.DiffDataset import load_OAT_CDiff
from tqdm.auto import tqdm, trange
import pickle
from FEA import *
from joblib import Parallel, delayed
torch.set_float32_matmul_precision('high')

torch.autograd.set_grad_enabled(False)

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="DOMTopoDiff", help="dataset name. Options: labeled, DOMTopoDiff")
args.add_argument("--dataset_path", type=str, default="Dataset", help="path to dataset. default Dataset")
args.add_argument("--jobs", type=int, default=1, help="number of jobs to run in parallel. default 1")
args.add_argument("--samples_path", type=str, default=None, help="path to samples. Must be provided.")
args.add_argument("--save_path", type=str, default="FEAResults", help="path to save results. default FEAResults")
args.add_argument("--save_name", type=str, default="FEA.pkl", help="name of the saved results. default FEA.pkl")

args = args.parse_args()

dataset = load_OAT_CDiff(latents_path=None,
                         data_path=args.dataset_path,
                         subset=args.dataset,
                         split='test',
                         unconditional_prob=0,
                         BC_dropout_prob=0,
                         C_dropout_prob=0)

with open(args.samples_path, 'rb') as f:
    results = pickle.load(f)

def get_CE_VFE(sample):
    
    top, vf, BCs, loads, shape, r = sample

    mesh = StructuredMesh2D(shape[0], shape[1], shape[0]/max(shape), shape[1]/max(shape))
    material = SingleMaterial(volume_fraction=vf, update_rule='OC', heavyside=False, penalty_schedule=None, penalty=3)
    kernel = StructuredStiffnessKernel(mesh)
    filter = StructuredFilter2D(mesh, 1.5)
    solver = CHOLMOD(kernel)
    optimizer = TopOpt(mesh, material, kernel, solver, filter, 10, ch_tol=0, fun_tol=0)

    optimizer.reset_BC()
    optimizer.reset_F()
    optimizer.add_BCs(BCs[:,0:2], BCs[:,2:].astype(np.bool_))
    optimizer.add_Forces(loads[:,0:2], loads[:,2:])

    try:
        compliance_gt = float(optimizer.FEA_integrals(np.array(top>0.5))[-1])
    
    except Exception as e:
        print(f"Error occurred while computing compliance_gt")
        n_samples = r.shape[0]
        
        compliance_gt = 0
        comps = np.zeros(n_samples) + 1000
        vfes = np.zeros(n_samples) + 1000
        CE = np.zeros(n_samples) + 1000
        return compliance_gt, comps, vfes, CE

    n_samples = r.shape[0]
    
    comps = np.zeros(n_samples)
    vfes = np.zeros(n_samples)
    for i in range(n_samples):
        try:
            comps[i] = float(optimizer.FEA_integrals(np.array(r[i].flatten()>0))[-1])
            vfes[i] = (r[i]>0).sum() / r[i].size
            vfes[i] -= vf
            vfes[i] = max(vfes[i] / vf, 0)
        except Exception as e:
            print(f"Error occurred while computing compliance for sample {i}")
            comps[i] = 1e12
            vfes[i] = 1e12
        
    CE = comps - compliance_gt
    CE /= compliance_gt
    
    return compliance_gt, comps, vfes, CE


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('processing samples...')
samples = []

for i in tqdm(range(len(results))):
    rnd_idx = i
    top = dataset.tensors[rnd_idx].numpy().flatten()
    shape = dataset.tensors[rnd_idx].numpy().shape[1:]
    BCs = dataset.BCs[0][rnd_idx]
    loads = dataset.BCs[1][rnd_idx]

    vf = dataset.Cs[0][rnd_idx]
    
    samples.append((top, vf, BCs, loads, shape, results[rnd_idx]))

if args.jobs > 1:
    print('processing samples in parallel...')
    fea_results = Parallel(n_jobs=args.jobs)(delayed(get_CE_VFE)(sample) for sample in tqdm(samples))
else:
    print('processing samples in serial...')
    fea_results = []
    prog = tqdm(samples)
    running_average = 0
    count = 0
    for sample in prog:
        fea_results.append(get_CE_VFE(sample))
        
        best_ce = fea_results[-1][3].min()
        
        if best_ce < 1:
            running_average = running_average + best_ce
            count += 1
            prog.set_description_str(f"Best CE: {best_ce*100:.4f} | Running Average: {running_average/count*100:.4f}")
        
        

with open(os.path.join(args.save_path, args.save_name), 'wb') as f:
    pickle.dump(fea_results, f)
print(f"Results saved to {os.path.join(args.save_path, args.save_name)}")

best_ce = []
best_vfe = []
mean_ce = []
mean_vfe = []
for i in range(len(fea_results)):
    best_id = np.argmin(fea_results[i][3])

    best_ce.append(fea_results[i][3][best_id])
    best_vfe.append(fea_results[i][2][best_id])
    
    mean_ce.append(np.mean(fea_results[i][3]))
    mean_vfe.append(np.mean(fea_results[i][2]))

best_ce = np.array(best_ce)
best_vfe = np.array(best_vfe)
mean_ce = np.array(mean_ce)
mean_vfe = np.array(mean_vfe)

keep = best_ce < 1
print((keep).sum(), "out of", len(best_ce), "samples are valid")
best_ce = best_ce[keep] * 100
best_vfe = best_vfe[keep] * 100

print((keep).sum(), "out of", len(mean_ce), "samples are valid")
mean_ce = mean_ce[keep] * 100
mean_vfe = mean_vfe[keep] * 100

print(f"Best CE: {np.mean(best_ce):.4f} ± {np.std(best_ce):.4f} | {np.median(best_ce):.4f}")
print(f"Best VFE: {np.mean(best_vfe):.4f} ± {np.std(best_vfe):.4f} | {np.median(best_vfe):.4f}")
print(f"Mean CE: {np.mean(mean_ce):.4f} ± {np.std(mean_ce):.4f} | {np.median(mean_ce):.4f}")
print(f"Mean VFE: {np.mean(mean_vfe):.4f} ± {np.std(mean_vfe):.4f} | {np.median(mean_vfe):.4f}")
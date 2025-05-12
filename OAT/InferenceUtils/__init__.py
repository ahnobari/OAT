import torch
from typing import Callable
from ..Models.NFAE import NFAE
from FEA import *
from ..DataUtils.DiffDataset import DiffDataset
import numpy as np
from tqdm.auto import trange

def INFD_Latent_Map(latent : torch.Tensor,
                    AE : NFAE,
                    coords : torch.Tensor,
                    cells : torch.Tensor,
                    inverse_normalize : Callable = None,
                    clip_to_range : bool = True,
                    clip_range : tuple = (-1, 1)):
    
    if inverse_normalize is not None:
        if clip_to_range:
            latent = torch.clamp(latent, clip_range[0], clip_range[1])
        latent = inverse_normalize(latent)
    
    
    # 1. Get the latent map
    pred_patch = AE.decoder(latent)
    pred_patch = AE.renderer(pred_patch,coords,cells)

    return pred_patch



def get_solver(top, vf, BCs, loads, shape):
    mesh = StructuredMesh2D(shape[0], shape[1], shape[0]/max(shape), shape[1]/max(shape))
    material = SingleMaterial(volume_fraction=vf, update_rule='OC', heavyside=False, penalty_schedule=None, penalty=3, void=1e-3)
    kernel = StructuredStiffnessKernel(mesh)
    filter = StructuredFilter2D(mesh, 1.5)
    solver = CHOLMOD(kernel)
    optimizer = TopOpt(mesh, material, kernel, solver, filter, 10, ch_tol=0, fun_tol=0, verbose=False)
    plotter = Plotter(optimizer)
    optimizer.plotter = plotter
    optimizer.gt_top = top
    optimizer.reset_BC()
    optimizer.reset_F()
    optimizer.add_BCs(BCs[:,0:2], BCs[:,2:].astype(np.bool_))
    optimizer.add_Forces(loads[:,0:2], loads[:,2:])
    
    return optimizer

class OptimizerGuidance:
    def __init__(self, dataset : DiffDataset, pre_process_solvers: bool = False):
        self.dataset = dataset
        self.pre_process_solvers = pre_process_solvers
        self.preprocessed = False
        
        if self.pre_process_solvers:
            self.solvers = [self._get_solver(i) for i in trange(len(self.dataset))]
            self.preprocessed = True
        else:
            self.solvers = None
        
    def _get_solver(self, idx):
        if self.preprocessed:
            return self.solvers[idx]
        
        top = self.dataset.tensors[idx].numpy().flatten()
        shape = self.dataset.tensors[idx].numpy().shape[1:]
        BCs = self.dataset.BCs[0][idx]
        loads = self.dataset.BCs[1][idx]

        vf = self.dataset.Cs[0][idx]
        
        return get_solver(top, vf, BCs, loads, shape)
    
    def make_guidance_function(self, idx, skip_init=3, n_solves=5, intermediate_null=False, intermediate_decay=1.0, binary_delta=False, normalize_grad=True, low_rank=False, tau=0.99):
        
        solver = self._get_solver(idx)
        
        def guidance_function(denoised, t, history, model, final_callable, step, total_steps, previous_grad, **kwargs):
            if step < skip_init:
                return torch.zeros_like(denoised.pred_original_sample)
            
            interval = int(np.round((total_steps - skip_init) / n_solves))
            
            if (step - skip_init) % interval != 0:
                if intermediate_null:
                    return torch.zeros_like(denoised.pred_original_sample)
                else:
                    return previous_grad * intermediate_decay
            
            pred = denoised.pred_original_sample.clone()
            pred = (pred + 1) / 2 * model.latent_scale - model.latent_shift
            
            pred_patch = final_callable(pred)
            shape = pred_patch.shape[2:]
            current_top = (pred_patch.detach().cpu().numpy().flatten() + 1) / 2
            current_top = current_top.astype(np.float64)
            solver.max_iter = 1
            stepped = solver.optimize(rho_init=current_top)[0]
            
            if binary_delta:
                epsilon = solver.material_model.void
                st_delta = current_top.reshape(shape) - (stepped.reshape(shape)<=(current_top.reshape(shape)-epsilon)) * current_top.reshape(shape) + (stepped.reshape(shape)>(current_top.reshape(shape)+epsilon))
                target = np.clip(st_delta, 0, 1) * 2 - 1
                target = torch.tensor(target).float().cuda()
            else:
                stepped = np.clip(stepped.reshape(shape), 0, 1) * 2 - 1
                target = torch.tensor(stepped).float().cuda()
                
            with torch.enable_grad():
                pred_param = denoised.pred_original_sample.clone()
                pred_param.requires_grad = True
                
                pred_param = (pred_param + 1) / 2 * model.latent_scale - model.latent_shift
                pred_param_patch = final_callable(pred_param).squeeze()
                loss = torch.nn.functional.l1_loss(pred_param_patch, target)
                
                grad = torch.autograd.grad(loss, pred_param)[0].detach().clone()
            
            if low_rank:
                Z_t = denoised.prev_sample.clone().squeeze()
                # svd
                U, S, V = torch.svd(Z_t)
                
                eig = S.square().flatten()
                c = torch.cumsum(eig, dim=0)/torch.sum(eig)
                r = torch.min(torch.where(c>=tau)[0])
                
                U_r = U[:, :r]
                V_r = V[:r, :]
                
                G = grad.squeeze()
                G_ = U_r.T @ G @ V_r.T
                grad = U_r @ G_ @ V_r
                grad = grad.reshape(denoised.pred_original_sample.shape)
            
            if normalize_grad:
                grad = grad / torch.norm(grad)
            
            return grad
        
        return guidance_function
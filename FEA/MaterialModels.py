import numpy as np
import warnings
from .mma import mmasub

class MaterialModel:
    def __init__(self, E=1.0, nu=0.33, n_sets=1):
        self.E = E
        self.nu = nu
        self.n_sets = n_sets

    def __call__(self, rho, iteration, **kwargs):
        return self.__forward__(rho, iteration, **kwargs)

    def evaluate_constraint(self, rho, dg, vol, **kwargs):
        return self.__con__(rho, dg, vol, **kwargs)

    def base_properties(self):
        return {"E": self.E, "nu": self.nu, "n_sets": self.n_sets}

    def grad(self, rho, iteration, **kwargs):
        return self.__backward__(rho, iteration, **kwargs)

    def is_terminal(self, iteration):
        return True

    def init_desvars(self, nel, **kwargs):
        return self.__initvars__(nel, **kwargs)
    
    def update_desvars(self, rho, df, dg, **kwargs):
        return self.__updatevars__(rho, df, dg, **kwargs)
    
    def find_threshold(self, rho, dg, V, **kwargs):
        return self.__th__(rho, dg, V, **kwargs)


class SingleMaterial(MaterialModel):
    def __init__(
        self,
        E=1.0,
        nu=0.33,
        void=1e-6,
        penalty=3.0,
        volume_fraction=0.25,
        penalty_schedule=None,
        update_rule='MMA',
        heavyside=True,
        beta=2,
        eta=0.5
    ):
        super().__init__(E=E, nu=nu, n_sets=1)
        self.void = void
        self.penalty = penalty
        self.volume_fraction = volume_fraction
        self.penalty_schedule = penalty_schedule
        self.update_rule = update_rule
        self.p_rho = None
        self.p_df = None
        self.heavyside = heavyside
        self.beta = beta
        self.eta = eta
        
        self.x_1 = None
        self.x_2 = None
        self.low = None
        self.upp = None
        
        if self.update_rule not in ['PGD', 'OC', 'MMA']:
            raise ValueError('update_rule must be either "PGD", "OC" or "MMA"')

    def __th__(self, rho, dg, V, np=np):
        lb = 0
        ub = 1
        for _ in range(1000):
            th = (lb + ub) / 2
            if ((rho.reshape(-1, 1)>th) * dg.reshape(-1, 1)).sum() > V * self.volume_fraction:
                lb = th
            else:
                ub = th
        return th
        
    
    def __forward__(self, rho, iteration, plain=False, np=np, **kwargs):
        if plain:
            rho = np.clip(rho, self.void, 1.0)
            return rho
        else:
            pen = self.penalty

            if self.penalty_schedule is not None:
                pen = self.penalty_schedule(self.penalty, iteration)

            if self.heavyside:
                rho = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            rho = rho**pen
            rho = np.clip(rho, self.void, 1.0)

            return rho

    def __backward__(self, rho, iteration, np=np, **kwargs):

        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, iteration)

        rho = np.clip(rho, self.void, 1.0)

        if self.heavyside:
            rho_heavy = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            df = pen * rho_heavy ** (pen - 1) * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
        else:
            df = pen * rho ** (pen - 1)

        return df

    def __con__(self, rho, dg, vol, np=np):
        V = (rho.squeeze() * dg.squeeze()).sum()

        if V / vol <= self.volume_fraction:
            return np.array([True])
        else:
            return np.array([False])

    def __ocP__(self, df, dg, rho, np=np):

        ocP = rho * np.nan_to_num(np.sqrt(np.maximum(-df / dg.reshape(-1, 1),0)), nan=0)

        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3

        return ocP

    def __initvars__(self, nel, np=np, dtype=np.float64):
        return np.ones([nel], dtype=dtype)

    def is_terminal(self, iteration):
        if self.penalty_schedule is not None:
            return self.penalty_schedule(self.penalty, iteration) == self.penalty
        else:
            return True

    def MMA_update(self, rho, df, dg, np=np, V=None, iteration=0, move =0.2, comp=1, **kwargs):
        
        if V is None:
            V = dg.sum()
        
        if self.x_1 is None or iteration == 0:
            self.x_1 = np.copy(rho)
            self.x_2 = np.copy(rho)
            self.low = np.ones([rho.shape[0], 1])
            self.upp = np.ones([rho.shape[0], 1])  
            self.comp_base = comp
        
        f_val  = ((rho * dg).sum() - self.volume_fraction*V)
        
        a = np.zeros([1,1])
        c = np.ones([1,1]) * 10000
        d = np.zeros([1,1])

        rho_new, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
            1,
            rho.shape[0],
            iteration+1, 
            np.copy(rho).reshape(-1,1),
            np.zeros_like(rho.reshape(-1,1)) + self.void,
            np.ones_like(rho.reshape(-1,1)),
            self.x_1.reshape(-1,1),
            self.x_2.reshape(-1,1),
            0,
            df.reshape(-1,1) / self.comp_base * 100,
            np.array([[f_val]]) / self.volume_fraction,
            np.ones_like(rho.reshape(1,-1)) * dg[0] / self.volume_fraction if dg.size == 1 else dg.reshape(1,-1),
            self.low,
            self.upp,
            1.0,
            a,
            c,
            d,
            np=np,
        )
        
        self.x_2 = np.copy(self.x_1)
        self.x_1 = np.copy(rho)
        
        return rho_new.reshape(-1)
        
    def OC_update(self, rho, df, dg, np=np, move=0.2, V=None, **kwargs):
        rho = rho.reshape(-1, 1)
        xU = np.clip(rho + move, 0.0, 1.0)
        xL = np.clip(rho - move, 0.0, 1.0)
        
        if V is None:
            V = dg.sum()

        ocP = self.__ocP__(df, dg, rho, np=np)
        ocP = np.maximum(1e-12, ocP)

        l1 = 1e-9
        l2 = 1e9

        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
        )

        while (l2 - l1) / (l2 + l1) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(
                0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
            )

            valids = self.evaluate_constraint(
                rho_new, dg, V, np=np
            )
            
            if valids[0]:
                l2 = lmid
            else:
                l1 = lmid
                
        return rho_new.reshape(-1)
    
    def PGD_update(self, rho, df, dg, np=np, res=0, V=None, iteration=0, comp=1, **kwargs):
        rho = rho.reshape(-1, 1)
        df = df.reshape(-1, 1)
        if V is None:
            V = dg.sum()
        
        if self.p_rho is None or res>1e-3:
            alpha = 1e-2/np.linalg.norm(df, ord=np.inf)
            self.alpha = alpha
            self.p_d = -np.copy(df)
            
            z = rho - self.alpha * df
        else:
            # armijo rule for step size
            # alpha = self.alpha
            # mul = 1.0
            # if comp > self.p_comp + 1e-4 * self.alpha * (-self.p_df.T @ self.p_df):
            #     mul = 0.5
            # elif df.T @ df < 0.1 * self.p_df.T @ self.p_df:
            #     mul = 2
            alpha = 1/(np.linalg.norm(df - self.p_df) + 1e-6) * np.linalg.norm(rho - self.p_rho)
            
            beta = max(df.T @ (df-self.p_df) / (self.p_df.T @ self.p_df),0)
            d = self.p_d*beta - df
            alpha = 1/(np.linalg.norm(d - self.p_d) + 1e-6) * np.linalg.norm(rho - self.p_rho)
            self.alpha = alpha 
            self.p_d = d
            
            z = rho + self.alpha * d
        
        self.p_rho = np.copy(rho)
        self.p_df = np.copy(df)
        self.p_comp = comp
        
        rho_ = z.flatten()
        
        if (np.clip(rho_, 0, 1)*dg).sum() <= V * self.volume_fraction:
            return np.clip(rho_, 0, 1)
        
        y_low = -np.max(rho_/dg)
        y_high = 1 - np.min(rho_/dg)
        for _ in range(1000):
            y_mid = (y_low + y_high) / 2.0
            x = np.clip(rho_ + y_mid*dg, 0, 1)
            diff = np.sum(x*dg) - V * self.volume_fraction
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                y_high = y_mid
            else:
                y_low = y_mid
        rho = np.clip(rho_ + y_mid*dg, 0, 1)
            
        return rho.reshape(-1)
    
    def __updatevars__(self, rho, df, dg, np=np, **kwargs):
        if self.update_rule == 'PGD':
            return self.PGD_update(rho, df, dg, np=np, **kwargs)
        elif self.update_rule == 'OC':
            return self.OC_update(rho, df, dg, np=np, **kwargs)
        else:
            return self.MMA_update(rho, df, dg, np=np, **kwargs)

class PenalizedMultiMaterial(MaterialModel):
    def __init__(
        self,
        E=1.0,
        nu=0.33,
        n_material=2,
        mass=np.array([[0.25], [0.25]]),
        E_mat=np.array([[1.0], [0.5]]),
        rho_mat=None,
        void=1e-6,
        penalty=3.0,
        penalty_schedule=None,
        update_rule='MMA',
        heavyside=True,
        beta=2,
        eta=0.5
    ):
        super().__init__(E=E, nu=nu, n_sets=n_material)
        self.n_material = n_material
        self.mass = mass
        self.E_mat = E_mat
        self.rho_mat = rho_mat
        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule
        self.update_rule = update_rule
        self.p_rho = None
        self.p_df = None
        self.heavyside = heavyside
        self.beta = beta
        self.eta = eta
        
        self.x_1 = None
        self.x_2 = None
        self.low = None
        self.upp = None
        
        if self.update_rule not in ['PGD', 'OC', 'MMA']:
            raise ValueError('update_rule must be either "PGD", "OC" or "MMA"')

        if self.E_mat.shape[0] != self.n_material:
            raise ValueError(
                "E_mat must have the same number of materials as n_material"
            )

        if self.rho_mat is not None:
            if self.rho_mat.shape[0] != self.n_material:
                raise ValueError(
                    "rho_mat must have the same number of materials as n_material"
                )

            if self.mass.shape[0] != 1:
                raise ValueError(
                    "if rho_mat is not None, mass must have only one value"
                )

            self.mass = self.mass.flatten()[0]
        else:
            if self.mass.shape[0] != self.n_material:
                raise ValueError(
                    "Mass and E_mat must have the same number of materials"
                )

    def __forward__(self, rho, iteration, plain=False, np=np):
        if plain:
            rho = rho @ np.array(self.E_mat)
            rho = np.clip(rho, self.void, 1.0)
            return rho.reshape(-1)
        else:
            pen = self.penalty

            if self.penalty_schedule is not None:
                pen = self.penalty_schedule(self.penalty, iteration)
            rho = np.clip(rho, self.void, 1.0)
            rho_ = rho**pen
            rho__ = 1 - rho_
            rho_ *= (
                rho__[
                    :,
                    np.where(~np.eye(self.n_material, dtype=bool))[1].reshape(
                        self.n_material, -1
                    ),
                ]
                .transpose(1, 0, 2)
                .prod(axis=-1)
                .T
            )

            E = rho_ @ np.array(self.E_mat)
            E = np.clip(E, self.void, np.inf)
            return E.reshape(-1)

    def __backward__(self, rho, iteration, np=np):

        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, iteration)

        rho = np.clip(rho, self.void, 1.0)
        rho_ = pen * rho ** (pen - 1)
        rho__ = 1 - rho**pen
        rho___ = rho**pen

        d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
        d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
        d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
        d = d.prod(axis=-1).transpose(0, 2, 1)

        mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
        mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

        d *= mul
        d = d @ np.array(self.E_mat)

        df_ = d.squeeze().T

        return df_

    def __con__(self, rho, dg, vol, np=np):
        if self.rho_mat is None:
            V = (rho * dg.reshape(-1, 1)).sum(axis=0)

            return (V / vol - np.array(self.mass).reshape(-1)) <= 0

        else:
            V = ((rho * dg.reshape(-1, 1)) @ self.rho_mat).sum()

            return np.ones([self.n_material], dtype=bool) * (
                (V / vol) <= np.array(self.mass)
            )

    def __ocP__(self, df, dg, rho, np=np):
        ocP = rho * np.nan_to_num(np.sqrt(-df / dg.reshape(-1, 1)), nan=0)

        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3

        return ocP

    def __initvars__(self, nel, np=np, dtype=np.float64):
        rho = np.ones([nel, self.n_material], dtype=dtype) * self.void
        return rho

    def is_terminal(self, iteration):
        if self.penalty_schedule is not None:
            return self.penalty_schedule(self.penalty, iteration) == self.penalty
        else:
            return True
        
    def OC_update(self, rho, df, dg, np=np, move=0.2, V=None, **kwargs):
        
        if V is None:
            V = dg.sum()
        
        xU = np.clip(rho + move, 0.0, 1.0)
        xL = np.clip(rho - move, 0.0, 1.0)

        ocP = self.ocP(df, dg, rho, np=np)
        ocP = np.maximum(1e-12, ocP)

        size = self.n_material
        l1 = 1e-9 * np.ones(size, dtype=rho.dtype)
        l2 = 1e9 * np.ones(size, dtype=rho.dtype)

        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
        )

        while np.any((l2 - l1) / (l2 + l1) > 1e-6):
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(
                0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
            )

            valids = self.evaluate_constraint(
                rho_new, dg, V, np=np
            )

            l2[valids] = lmid[valids]
            l1[~valids] = lmid[~valids]
            
        return rho_new
    
    def PGD_update(self, rho, df, dg, np=np, res=0, V=None, **kwargs):
        
        if V is None:
            V = dg.sum()
    
        if self.p_rho is None:
            alpha = 1/np.linalg.norm(df)
        else:
            alpha = 1/(np.linalg.norm(df - self.p_df, axis=0) + 1e-6) * np.linalg.norm(rho - self.p_rho, axis=0)
            alpha = alpha.reshape(1,-1)
        
        self.p_rho = np.copy(rho)
        self.p_df = np.copy(df)
        
        z = rho - alpha * df
        
        size = self.n_material
        
        for sp in range(size):
            rho_ = z[:, sp]
            if (np.clip(rho_, 0, 1)*dg).sum() <= V * self.mass[sp,0]:
                rho[:, sp] = np.clip(rho_, 0, 1)
            else:
                y_low = -np.max(rho_/dg)
                y_high = 1 - np.min(rho_/dg)
                for _ in range(10000):
                    y_mid = (y_low + y_high) / 2.0
                    x = np.clip(rho_ + y_mid*dg, 0, 1)
                    diff = np.sum(x*dg) - V * self.mass[sp,0]
                    if abs(diff) < 1e-12:
                        break
                    if diff > 0:
                        y_high = y_mid
                    else:
                        y_low = y_mid
                rho[:, sp] = np.clip(rho_ + y_mid*dg, 0, 1)
                
        return rho
    
    def MMA_update(self, rho, df, dg, np=np, V=None, iteration=0, move =0.2, comp=1, **kwargs):
        
        if V is None:
            V = dg.sum()
        
        if self.x_1 is None or iteration == 0:
            self.x_1 = np.copy(rho)
            self.x_2 = np.copy(rho)
            self.low = np.ones_like(rho).reshape(-1,1)
            self.upp = np.ones_like(rho).reshape(-1,1)
            self.comp_base = comp
            
        size = self.n_material
        f_val  = ((rho * dg.reshape(-1, 1)).sum(axis=0) - np.array(self.mass).reshape(-1)*V)
        
        a = np.zeros([size,1])
        c = np.ones([size,1]) * 100000
        d = np.zeros([size,1])
        
        con_scale = np.array(self.mass).min()
        n = rho.shape[0]
        dG = np.zeros([size,rho.size])
        for i in range(size):
            dG[i,i*n:(i+1)*n] = dg[0]/con_scale * size if dg.size == 1 else dg.reshape(-1)/ con_scale * size
        
        rho_new, _, _, _, _, _, _, _, _, self.low, self.upp = mmasub(
            size,
            rho.size,
            iteration+1, 
            np.copy(rho).reshape(-1,1),
            np.zeros_like(rho).reshape(-1,1),
            np.ones_like(rho).reshape(-1,1),
            self.x_1.reshape(-1,1),
            self.x_2.reshape(-1,1),
            0,
            df.reshape(-1,1)/self.comp_base * 1000,
            f_val.reshape(-1,1) / con_scale * size,
            dG,
            self.low,
            self.upp,
            1.0,
            a,
            c,
            d,
            np=np,
        )
        
        self.x_2 = np.copy(self.x_1)
        self.x_1 = np.copy(rho)
        
        return rho_new.reshape(-1,size)
    
    def __updatevars__(self, rho, df, dg, np=np, **kwargs):
        if self.update_rule == 'PGD':
            return self.PGD_update(rho, df, dg, np=np, **kwargs)
        elif self.update_rule == 'MMA':
            return self.MMA_update(rho, df, dg, np=np, **kwargs)
        else:
            return self.OC_update(rho, df, dg, np=np, **kwargs)
        
import itertools
import math

import dgl
import ot
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return - sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean(
        torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma),
                    min=0)) + \
           torch.mean(
               torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma),
                           min=0))
    return loss

def pairwise_distances(receptor_coords: torch.tensor, ligand_coords: torch.tensor) -> torch.tensor:
    """Returns matrix of pairwise Euclidean distances.
    Parameters
    ----------
    coords1: torch.tensor
        A torch tensor of shape `(N, 3)`
    coords2: torch.tensor
        A torch tensor of shape `(M, 3)`
    Returns
    -------
    torch.tensor
        A `(N,M)` array with pairwise distances.
  """
    "TODO: this should check the dimension"
    return torch.sum((receptor_coords.view(1, -1, 3) - ligand_coords.view(-1, 1, 3)) ** 2, dim=-1) ** 0.5

def cutoff_filter(d: torch.tensor, x: torch.tensor, cutoff=8.0) -> torch.tensor:
    """Applies a cutoff filter on pairwise distances
    Parameters
    ----------
    d: torch.tensor
        Pairwise distances matrix. A torch tensor of shape `(N, M)`
    x: torch.tensor
        Matrix of shape `(N, M)`
    cutoff: float, optional (default 8)
        Cutoff for selection in Angstroms
    Returns
    -------
    torch.tensor
        A `(N,M)` array with values where distance is too large thresholded to 0.
    """
    return torch.where(d < cutoff, x, torch.zeros_like(x))

def vina_nonlinearity(c: torch.tensor, w: float, Nrot: int) -> torch.tensor:
    """Computes non-linearity used in Vina.
    Parameters
    ----------
    c: torch.tensor
        A torch tensor of shape `(N, M)`
    w: float
        Weighting term
    Nrot: int
        Number of rotatable bonds in this molecule
    Returns
    -------
    torch.tensor
        A `(N, M)` array with activations under a nonlinearity.
    """
    return c / (1 + w * Nrot)

def vina_repulsion(d: torch.tensor) -> torch.tensor:
    """Computes Autodock Vina's repulsion interaction term.
    Parameters
    ----------
    d: torch.tensor
        A torch tensor of shape `(N, M)`.
    Returns
    -------
    torch.tensor
        A `(N, M)` array with repulsion terms.
    """
    return torch.where(d < 0, d**2, torch.zeros_like(d))

def vina_hydrophobic(d: torch.tensor) -> torch.tensor:
    """Computes Autodock Vina's hydrophobic interaction term.
    Here, d is the set of surface distances as defined in [1]_
    Parameters
    ----------
    d: torch.tensor
        A torch tensor of shape `(N, M)`.
    Returns
    -------
    torch.tensor
        A `(N, M)` array of hydrophoboic interactions in a piecewise linear curve.
    References
    ----------
    .. [1] Jain, Ajay N. "Scoring noncovalent protein-ligand interactions:
        a continuous differentiable function tuned to compute binding affinities."
        Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    return torch.where(d < 0.5, torch.ones_like(d), torch.where(d < 1.5, 1.5 - d, torch.zeros_like(d)))


def vina_hbond(d: torch.tensor) -> torch.tensor:
    """Computes Autodock Vina's hydrogen bond interaction term.
    Here, d is the set of surface distances as defined in [1]_
    Parameters
    ----------
    d: torch.tensor
        A torch tensor of shape `(N, M)`.
    Returns
    -------
    torch.tensor
        A `(N, M)` array of hydrophoboic interactions in a piecewise linear curve.
    References
    ----------
    .. [1] Jain, Ajay N. "Scoring noncovalent protein-ligand interactions:
        a continuous differentiable function tuned to compute binding affinities."
        Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    return torch.where(d < -0.7, torch.ones_like(d), torch.where(d < 0, (1.0 / 0.7) * (0 - d), torch.zeros_like(d)))


def vina_gaussian_first(d: torch.tensor) -> torch.tensor:
    """Computes Autodock Vina's first Gaussian interaction term.
    Here, d is the set of surface distances as defined in [1]_
    Parameters
    ----------
    d: torch.tensor
        A torch tensor of shape `(N, M)`.
    Returns
    -------
    torch.tensor
        A `(N, M)` array of gaussian interaction terms.
    References
    ----------
    .. [1] Jain, Ajay N. "Scoring noncovalent protein-ligand interactions:
        a continuous differentiable function tuned to compute binding affinities."
        Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
  
    return torch.exp(-(d / 0.5)**2)


def vina_gaussian_second(d: torch.tensor) -> torch.tensor:
    """Computes Autodock Vina's second Gaussian interaction term.
    Here, d is the set of surface distances as defined in [1]_
    Parameters
    ----------
    d: torch.tensor
        A torch tensor of shape `(N, M)`.
    Returns
    -------
    torch.tensor
        A `(N, M)` array of gaussian interaction terms.
    References
    ----------
    .. [1] Jain, Ajay N. "Scoring noncovalent protein-ligand interactions:
        a continuous differentiable function tuned to compute binding affinities."
        Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    return torch.exp(-((d - 3) / 2)**2)


def weighted_linear_sum(w: torch.tensor, x: torch.tensor) -> torch.tensor:
    """Computes weighted linear sum.
    Parameters
    ----------
    w: torch.tensor
        A torch tensor of shape `(N,)`
    x: torch.tensor
        A torch tensor of shape `(N, M, L)`
    Returns
    -------
    torch.tensor
        A torch tensor of shape `(M, L)`
    """
    return torch.tensordot(w, x, axes=1)


def compute_vina_energy(coords1: torch.tensor, coords2: torch.tensor,
                     weights: torch.tensor, wrot: float, Nrot: int) -> torch.tensor:
    """Computes the Vina Energy function for two molecular conformations
    Parameters
    ----------
    coords1: torch.tensor
        Molecular coordinates of shape `(N, 3)`
    coords2: torch.tensor
        Molecular coordinates of shape `(M, 3)`
    weights: torch.tensor
        A torch tensor of shape `(5,)`. The 5 values are weights for repulsion interaction term,
        hydrophobic interaction term, hydrogen bond interaction term,
        first Gaussian interaction term and second Gaussian interaction term.
    wrot: float
        The scaling factor for nonlinearity
    Nrot: int
        Number of rotatable bonds in this calculation
    Returns
    -------
    torch.tensor
        A scalar value with free energy
    """
  # TODO(rbharath): The autodock vina source computes surface distances
  # which take into account the van der Waals radius of each atom type.
    dists = pairwise_distances(coords1, coords2)
    repulsion = vina_repulsion(dists)
    hydrophobic = vina_hydrophobic(dists)
    hbond = vina_hbond(dists)
    gauss_1 = vina_gaussian_first(dists)
    gauss_2 = vina_gaussian_second(dists)

    # Shape (N, M)
    interactions = weighted_linear_sum(weights, torch.tensor([repulsion, hydrophobic, hbond, gauss_1, gauss_2]))

    # Shape (N, M)
    thresholded = cutoff_filter(dists, interactions)

    free_energies = vina_nonlinearity(thresholded, wrot, Nrot)
    return torch.sum(free_energies)

def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached


def compute_revised_intersection_loss(lig_coords, rec_coords, alpha = 0.2, beta=8, aggression=0):
    distances = compute_sq_dist_mat(lig_coords,rec_coords)
    if aggression > 0:
        aggression_term = torch.clamp(-torch.log(torch.sqrt(distances)/aggression+0.01), min=1)
    else:
        aggression_term = 1
    distance_losses = aggression_term * torch.exp(-alpha*distances * torch.clamp(distances*4-beta, min=1))
    return distance_losses.sum()

class BindingLoss(_Loss):
    def __init__(self, vina_energy_loss_weight=0, ot_loss_weight=1, intersection_loss_weight=0, intersection_sigma=0, geom_reg_loss_weight=1, loss_rescale=True,
                 intersection_surface_ct=0, key_point_alignmen_loss_weight=0,revised_intersection_loss_weight=0, centroid_loss_weight=0, kabsch_rmsd_weight=0,translated_lig_kpt_ot_loss=False, revised_intersection_alpha=0.1, revised_intersection_beta=8, aggression=0) -> None:
        super(BindingLoss, self).__init__()
        self.vina_energy_loss_weight = vina_energy_loss_weight
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.intersection_sigma = intersection_sigma
        self.revised_intersection_loss_weight =revised_intersection_loss_weight
        self.intersection_surface_ct = intersection_surface_ct
        self.key_point_alignmen_loss_weight = key_point_alignmen_loss_weight
        self.centroid_loss_weight = centroid_loss_weight
        self.translated_lig_kpt_ot_loss= translated_lig_kpt_ot_loss
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.revised_intersection_alpha = revised_intersection_alpha
        self.revised_intersection_beta = revised_intersection_beta
        self.aggression =aggression
        self.loss_rescale = loss_rescale
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.mse_loss = MSELoss()

    def forward(self, ligs_coords, recs_coords, ligs_coords_pred, ligs_pocket_coords, recs_pocket_coords, ligs_keypts,
                recs_keypts, rotations, translations, geom_reg_loss, device, **kwargs):
        # Compute MSE loss for each protein individually, then average over the minibatch.
        ligs_coords_loss = 0
        recs_coords_loss = 0
        vina_energy_loss = 0
        ot_loss = 0
        intersection_loss = 0
        intersection_loss_revised = 0
        keypts_loss = 0
        centroid_loss = 0
        kabsch_rmsd_loss = 0

        for i in range(len(ligs_coords_pred)):
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            ligs_coords_loss = ligs_coords_loss + self.mse_loss(ligs_coords_pred[i], ligs_coords[i])

            if self.ot_loss_weight > 0:
                # Compute the OT loss for the binding pocket:
                ligand_pocket_coors = ligs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes
                receptor_pocket_coors = recs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes

                ## (N, K) cost matrix
                if self.translated_lig_kpt_ot_loss:
                    cost_mat_ligand = compute_sq_dist_mat(receptor_pocket_coors, (rotations[i] @ ligs_keypts[i].t()).t() + translations[i] )
                else:
                    cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligs_keypts[i])
                cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, recs_keypts[i])

                ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, device)
                ot_loss += ot_dist
            if self.key_point_alignmen_loss_weight > 0:
                keypts_loss += self.mse_loss((rotations[i] @ ligs_keypts[i].t()).t() + translations[i],
                                             recs_keypts[i])

            if self.intersection_loss_weight > 0:
                intersection_loss = intersection_loss + compute_body_intersection_loss(ligs_coords_pred[i],
                                                                                       recs_coords[i],
                                                                                       self.intersection_sigma,
                                                                                       self.intersection_surface_ct)
            if self.vina_energy_loss_weight > 0:
                vina_energy_loss = vina_energy_loss + compute_vina_energy(ligs_coords_pred[i], 
                                                                          recs_coords[i], 
                                                                          weights=None, 
                                                                          wrot=None, Nrot=None)


            if self.revised_intersection_loss_weight > 0:
                intersection_loss_revised = intersection_loss_revised + compute_revised_intersection_loss(ligs_coords_pred[i],
                                                                                       recs_coords[i], alpha=self.revised_intersection_alpha, beta=self.revised_intersection_beta, aggression=self.aggression)

            if self.kabsch_rmsd_weight > 0:
                lig_coords_pred = ligs_coords_pred[i]
                lig_coords = ligs_coords[i]
                lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
                lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

                A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

                U, S, Vt = torch.linalg.svd(A)

                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                kabsch_rmsd_loss += self.mse_loss((rotation @ lig_coords.t()).t() + translation, lig_coords_pred)

            centroid_loss += self.mse_loss(ligs_coords_pred[i].mean(dim=0), ligs_coords[i].mean(dim=0))

        if self.loss_rescale:
            ligs_coords_loss = ligs_coords_loss / float(len(ligs_coords_pred))
            ot_loss = ot_loss / float(len(ligs_coords_pred))
            intersection_loss = intersection_loss / float(len(ligs_coords_pred))
            keypts_loss = keypts_loss / float(len(ligs_coords_pred))
            centroid_loss = centroid_loss / float(len(ligs_coords_pred))
            kabsch_rmsd_loss = kabsch_rmsd_loss / float(len(ligs_coords_pred))
            intersection_loss_revised = intersection_loss_revised / float(len(ligs_coords_pred))
            geom_reg_loss = geom_reg_loss / float(len(ligs_coords_pred))

        loss = self.vina_energy_loss_weight * vina_energy_loss + ligs_coords_loss + self.ot_loss_weight * ot_loss + self.intersection_loss_weight * intersection_loss + keypts_loss * self.key_point_alignmen_loss_weight + centroid_loss * self.centroid_loss_weight + kabsch_rmsd_loss * self.kabsch_rmsd_weight + intersection_loss_revised *self.revised_intersection_loss_weight + geom_reg_loss*self.geom_reg_loss_weight
        return loss, {'ligs_coords_loss': ligs_coords_loss, 'recs_coords_loss': recs_coords_loss, 'ot_loss': ot_loss, 'vina_energy_loss': vina_energy_loss,
                      'intersection_loss': intersection_loss, 'keypts_loss': keypts_loss, 'centroid_loss:': centroid_loss, 'kabsch_rmsd_loss': kabsch_rmsd_loss, 'intersection_loss_revised': intersection_loss_revised, 'geom_reg_loss': geom_reg_loss}

class TorsionLoss(_Loss):
    def __init__(self) -> None:
        super(TorsionLoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, angles_pred, angles, masks, **kwargs):
        return self.mse_loss(angles_pred*masks,angles*masks)
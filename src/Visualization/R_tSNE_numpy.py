import numpy as np
from scipy.sparse import csr_matrix

from .utils_numpy import (
    norm_SPD,
    retraction,
    pairwise_airm_dist_2x2,
    pga,
    sqrt_inv_sqrt_2x2,
    log_2x2,
)
from pyriemann.utils.base import logm, invsqrtm
from pyriemann.utils.tangentspace import upper
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import pairwise_distance

from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.neighbors import NearestNeighbors
from scipy.special import xlogy

import matplotlib.pyplot as plt
import warnings
from time import time


class Riemannian_tSNE:
    """Riemannian version of t-SNE algorithm. Reduces a set of c x c SPD matrices into a set of 2 x 2 SPD matrices.

    Parameters
    ----------
    perplexity : int, default = None
        Perplexity used in the Riemannian t-SNE algorithm. If perplexity = None, it will be set to 0.75*n_matrices
    verbosity : int, default = 1
        Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.
    max_it : int, default = 10000
        Maximum number of iterations used for the Riemannian gradient descent.
    max_time : int, default = 300
        Maximum time on the run time of the Riemannian gradient descent in seconds.
    init : str, default = "random"
        The initialization method for the embedding. If "random", the embedding is randomly initialized.

    Attributes
    ----------
    res_opti : ndarray, shape (n_matrices, 2, 2)
        The result of the optimization process.
    """

    def __init__(
        self,
        perplexity=None,
        verbosity=1,
        max_it=10000,
        max_time=300,
        init="pga",
        seed=None,
        ann=False,
    ):
        self.perplexity = perplexity
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.init = init
        self.seed = seed
        self.ann = ann
        self.loss_evolution = None
        self.res_opti = None
        self.n_iterations = None
        self.runtime = None
        self.final_loss = None

    def compute_similarities(self, X):
        """Computed the high dimensional symmetrized conditional similarities p_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices to reduce.

        Returns
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        """
        if self.ann:
            return self._compute_similarities_ann(X)

        nb_mat = X.shape[0]
        Dsq = pairwise_distance(X, squared=True).astype(np.float32, copy=False)
        conditional_P = _binary_search_perplexity(Dsq, self.perplexity, 0)
        P = conditional_P + conditional_P.T
        return P / (2 * nb_mat)

    def _compute_similarities_ann(self, X):
        """Approximate compute_similarities using a tangent-space ANN index.

        Projects SPD matrices to the tangent space at the Fréchet mean
        (a Euclidean space), builds a nearest-neighbor index there, then
        computes exact Riemannian distances only for the k ~ 3*perplexity
        nearest neighbors of each point.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)

        Returns
        -------
        P : ndarray, shape (n_matrices, n_matrices)
        """
        n = X.shape[0]
        k = min(n - 1, max(1, int(3 * self.perplexity)))

        # --- Step 1: embed into tangent space at Fréchet mean ---------------
        # Each SPD matrix maps to a symmetric matrix via
        #   S[i] = logm(mean^{-1/2} @ X[i] @ mean^{-1/2})
        # Flatten upper triangle → Euclidean vector
        mean = mean_covariance(X, metric="riemann")
        mean_invsqrt = invsqrtm(mean)
        conj = mean_invsqrt @ X @ mean_invsqrt  # (n, d, d)
        tangent = logm(conj)  # (n, d, d)
        embedding = upper(tangent)  # (n, d*(d+1)/2)

        # --- Step 2: k-NN in tangent space ----------------------------------
        # Request k+1 neighbors because kneighbors includes the query point itself
        nn_index = NearestNeighbors(
            n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        nn_index.fit(embedding)
        _, neighbors = nn_index.kneighbors(embedding)  # (n, k+1)
        neighbors = neighbors[:, 1:]  # (n, k) — drop self

        # --- Step 3: exact Riemannian distances for candidate pairs ---------
        # For each i, compute IS_i = invsqrtm(X[i]) once, then form
        #   M[j] = IS_i @ X[neighbors[i,j]] @ IS_i  for all k neighbors at once
        # and call batch eigvalsh on the (k, d, d) result.
        X_invsqrt = invsqrtm(X)  # (n, d, d)
        distances_sq = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            IS_i = X_invsqrt[i]  # (d, d)
            k_mats = X[neighbors[i]]  # (k, d, d)
            M_i = IS_i @ k_mats @ IS_i  # (k, d, d)
            eigvals = np.linalg.eigvalsh(M_i)  # (k, d)
            log_eig = np.log(np.maximum(eigvals, 1e-300))
            distances_sq[i] = np.sum(log_eig**2, axis=1).astype(np.float32)

        # --- Step 4: perplexity-matching ------------------------------------
        conditional_P = _binary_search_perplexity(distances_sq, self.perplexity, 0)

        # --- Step 5: build sparse P, symmetrize, normalise -----------------
        P_sparse = csr_matrix(
            (
                conditional_P.ravel(),
                neighbors.ravel(),
                np.arange(0, n * k + 1, k),
            ),
            shape=(n, n),
        )
        P_sparse = P_sparse + P_sparse.T
        P_sparse /= P_sparse.sum()
        return P_sparse.toarray()

    def compute_low_affinities(self, Y):
        """Computed the low dimensional similarities q_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, 2, 2)
            Set of SPD matrices.

        Returns
        ----------
        Q : ndarray, shape (n_matrices, n_matrices)
            The matrix of the low dimensional similarities conditional probabilities of Y.
        Dsq : ndarray, shape (n_matrices, n_matrices)
            The array containing the squared Riemannian distances between the points in X.
        """
        Dsq = pairwise_airm_dist_2x2(Y)
        inv_dist = 1.0 / (1.0 + Dsq)
        denominator = np.sum(inv_dist) - np.trace(inv_dist)
        Q = inv_dist / denominator
        np.fill_diagonal(Q, 0)
        return Q, Dsq

    def cost(self, P, Q):
        """Computed the loss of the t-SNE, that is the Kullback-Leibler divergence between P and Q.

        Parameters
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        Q : ndarray, shape (n_matrices, n_matrices)
            The matrix of the low dimensional similarities conditional probabilities of Y.

        Returns
        ----------
        _ : float
            The cost of the t-SNE.
        """
        # xlogy(a, b) = a * log(b), with xlogy(0, 0) = 0 — handles sparse P
        eye = np.eye(P.shape[0])
        return np.sum(xlogy(P, P + eye) - xlogy(P, Q + eye))

    def riemannian_gradient(self, Y, P, Q, Dsq):
        """Computed the Riemannian gradient of the loss of the t-SNE.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, 2, 2)
            Set of SPD matrices.
        P : ndarray, shape (n_matrices, n_matrices)
            The matrices of the symmetrized conditional probabilities of X.
        Q : ndarray, shape (n_matrices, n_matrices)
             The matrices of the low dimensional similarities conditional probabilities of Y.
        Dsq : ndarray, shape (n_matrices, n_matrices)
            The Riemannian distance matrix of Y.
        Returns
        ----------
        grad : ndarray, shape (n_matrices, 2, 2)
            The Riemannian gradient of the cost of the t-SNE.
        """
        nb_mat = P.shape[0]
        grad = np.zeros((nb_mat, 2, 2))
        Y_i_sqrt, Y_i_invsqrt = sqrt_inv_sqrt_2x2(Y)
        for i in range(nb_mat):
            log_riemann = (
                Y_i_sqrt[i] @ log_2x2(Y_i_invsqrt[i] @ Y @ Y_i_invsqrt[i]) @ Y_i_sqrt[i]
            )
            grad[i] = -4 * np.sum(
                ((P[i] - Q[i]) / (1 + Dsq[i]))[:, np.newaxis, np.newaxis] * log_riemann,
                axis=0,
            )
        return grad

    def riemannian_gradient_2(self, Y, P, Q, Dsq):
        nb_mat = P.shape[0]
        Y_sqrt, Y_invsqrt = sqrt_inv_sqrt_2x2(Y)  # both (n, 2, 2)

        # M[i,j] = Y_invsqrt[i] @ Y[j] @ Y_invsqrt[i]  →  (n, n, 2, 2)
        conj = Y_invsqrt[:, None] @ Y[None, :] @ Y_invsqrt[:, None]

        # Batch log of all n² matrices
        log_conj = log_2x2(conj.reshape(-1, 2, 2)).reshape(nb_mat, nb_mat, 2, 2)

        # log_riemann[i,j] = Y_sqrt[i] @ log_conj[i,j] @ Y_sqrt[i]
        log_riemann = Y_sqrt[:, None] @ log_conj @ Y_sqrt[:, None]

        # weights (n, n)
        W = (P - Q) / (1.0 + Dsq)

        # grad[i] = -4 * sum_j W[i,j] * log_riemann[i,j]
        grad = -4.0 * np.einsum("ij,ijkl->ikl", W, log_riemann)

        return grad

    def run_minimization(self, P):
        """Run the minimization to solve the t-SNE optimization.

        Parameters
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.

        Returns
        ----------
        current_sol : ndarray, shape (n_matrices, 2, 2)
            The solution of the t-SNE problem.
        """
        tol_step = 1e-6
        current_sol = self.initial_point
        self.loss_evolution = []
        initial_time = time()

        # loop over iterations
        for i in range(self.max_it):
            if self.verbosity >= 2 and i % 100 == 0:
                print("Iteration : ", i)

            # get the current value for the loss function
            Q, Dsq = self.compute_low_affinities(current_sol)
            loss = self.cost(P, Q)
            self.loss_evolution.append(loss)

            # get the direction of steepest descent
            # direction = self.riemannian_gradient(current_sol, P, Q, Dsq)
            direction = self.riemannian_gradient_2(current_sol, P, Q, Dsq)
            norm_direction = norm_SPD(current_sol, direction)

            # backtracking line search
            if i == 0:
                alpha = 1.0 / norm_direction
            else:
                # Pick initial step size based on where we were last time and look a bit further
                # See Boumal, 2023, Section 4.3 for more insights.
                alpha = 4 * (self.loss_evolution[-2] - loss) / (norm_direction**2)

            tau = 0.50
            r = 1e-4
            maxiter_linesearch = 25

            retracted = retraction(current_sol, -alpha * direction)
            Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
            loss_retracted = self.cost(P, Q_retracted)

            # Backtrack while the Armijo criterion is not satisfied
            for _ in range(maxiter_linesearch):
                if loss - loss_retracted > r * alpha * norm_direction**2:
                    break
                alpha = tau * alpha

                retracted = retraction(current_sol, -alpha * direction)
                Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
                loss_retracted = self.cost(P, Q_retracted)
            else:
                print("Maximum iteration in linesearched reached.")

            # update variable for next iteration
            current_sol = retracted

            # test if the step size is small
            crit = norm_SPD(current_sol, -alpha * direction)
            if crit <= tol_step:
                print("Min stepsize reached")
                break

            # test if the maximum time has been reached
            if time() - initial_time >= self.max_time:
                warnings.warn("Time limite reached after " + str(i) + " iterations.")
                break

        else:
            warnings.warn("Maximum iterations reached.")
        self.n_iterations = i
        self.final_loss = loss
        print(
            f"Optimization done in {time() - initial_time:.2f} seconds and with {i} iterations."
        )
        print(f"Final loss: {loss}")
        return current_sol

    def fit(self, X):
        """Fit X to 2x2 SDP matrices using the Riemannian t-SNE algorithm.

        Parameters
        ----------
        X : array_like of shape (n_matrices, n_channels, n_channels)

        Returns
        ----------
        res_opti :  ndarray, shape (n_matrices, 2, 2)
            The solution of the t-SNE problem.
        """
        fit_start = time()
        n_matrices, _, _ = X.shape

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_matrices)

        # Compute similarities in the high dimension space
        P = self.compute_similarities(X)

        if self.init == "random":
            # Sample SPD matrices near identity via Cholesky factorization
            sigma = 1e-2
            rng = np.random.default_rng(self.seed)
            L = np.zeros((n_matrices, 2, 2))
            L[:, 0, 0] = np.exp(rng.normal(0, sigma, n_matrices))
            L[:, 1, 0] = rng.normal(0, sigma, n_matrices)
            L[:, 1, 1] = np.exp(rng.normal(0, sigma, n_matrices))
            self.initial_point = L @ np.swapaxes(L, -1, -2)
        elif self.init == "pga":
            self.initial_point = pga(X, random_state=self.seed)
        else:
            raise ValueError("Unknown initialization method: " + str(self.init))
        if self.verbosity >= 1:
            print("Optimizing...")
        self.res_opti = self.run_minimization(P)
        self.runtime = time() - fit_start

        return self.res_opti

    def plot_loss(self):
        """Plot the evolution of the loss during the Riemannian gradient descent in log-log scale."""

        if self.loss_evolution is None:
            raise Exception("You need to fit the T-SNE before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.loss_evolution)
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()

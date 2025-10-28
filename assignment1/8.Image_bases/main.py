# task8_image_bases.py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

# ---------------------------
# Utility functions: handling MATLAB cell arrays
# ---------------------------
def _from_mat_cell(obj, idx):
    """
    Extract an element from a MATLAB cell array loaded by scipy.io.loadmat.
    """
    arr = obj
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return arr.flat[idx]
    return arr

def _to_np_stack(cell_item) -> np.ndarray:
    """
    Convert a cell element to a numpy array with shape (H, W, N).
    Cast to float32 for numerical stability.
    """
    A = np.array(cell_item)
    if A.ndim != 3:
        raise ValueError(f"Expected 3-D image stack, got shape {A.shape}")
    return A.astype(np.float32)

def _to_np_basis(cell_item) -> np.ndarray:
    """
    Convert a basis (19x19x4) to a numpy array with shape (H, W, K).
    """
    B = np.array(cell_item)
    if B.ndim != 3:
        raise ValueError(f"Expected 3-D basis, got shape {B.shape}")
    return B.astype(np.float32)

# ---------------------------
# Linear algebra core
# ---------------------------
def flatten_stack(stack_hwN: np.ndarray) -> np.ndarray:
    """
    Flatten an image stack (H,W,N) into (D,N), where D=H*W, row-major order.
    """
    H, W, N = stack_hwN.shape
    return stack_hwN.reshape(H*W, N, order="C")

def flatten_basis(basis_hwk: np.ndarray) -> np.ndarray:
    """
    Flatten basis images (H,W,K) into (D,K).
    """
    H, W, K = basis_hwk.shape
    return basis_hwk.reshape(H*W, K, order="C")

def orthonormalize_columns(E: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Orthonormalize the columns of E (D,K) via QR decomposition, ensuring the basis
    is numerically orthogonal. Returns Q such that Q^T Q = I.
    """
    Q, R = np.linalg.qr(E)
    # Fix column signs for consistency (optional)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q @ np.diag(diag_sign)
    return Q

def project_onto_basis(u: np.ndarray, E: np.ndarray, assume_orthonormal: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Project a single vector u (D,) onto the column space of E (D,K).
    Returns:
      up  : projection (D,)
      err : residual norm ||u - up||_2
      x   : coefficient vector (K,)
    If assume_orthonormal=False, solve by least squares.
    """
    if assume_orthonormal:
        # Orthonormal basis: x = E^T u ; up = E x
        x = E.T @ u
        up = E @ x
    else:
        # Non-orthogonal basis: solve min_x ||E x - u||_2
        x, *_ = np.linalg.lstsq(E, u, rcond=None)
        up = E @ x
    err = np.linalg.norm(u - up)
    return up, err, x

def batch_mean_error(stack_hwN: np.ndarray, basis_hwk: np.ndarray, ensure_orthonormal: bool = True) -> Tuple[float, List[float]]:
    """
    Compute the mean error of projecting an image stack onto a given basis.
    Returns: mean error value and the list of per-image errors.
    """
    U = flatten_stack(stack_hwN)       # (D,N)
    E = flatten_basis(basis_hwk)       # (D,K)
    if ensure_orthonormal:
        E = orthonormalize_columns(E)  # safeguard against numerical non-orthogonality

    errs = []
    for n in range(U.shape[1]):
        _, err, _ = project_onto_basis(U[:, n], E, assume_orthonormal=True)
        errs.append(float(err))
    return float(np.mean(errs)), errs

# ---------------------------
# Plotting helpers
# ---------------------------
def show_examples(stack_hwN: np.ndarray, title: str, n_show: int = 6):
    H, W, N = stack_hwN.shape
    n_show = min(n_show, N)
    cols = 3
    rows = int(np.ceil(n_show / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n_show):
        plt.subplot(rows, cols, i+1)
        plt.imshow(stack_hwN[:, :, i], cmap="gray")
        plt.axis("off")
        plt.title(f"#{i+1}")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def show_basis(basis_hwk: np.ndarray, title: str):
    H, W, K = basis_hwk.shape
    plt.figure(figsize=(3*K, 3))
    for k in range(K):
        plt.subplot(1, K, k+1)
        plt.imshow(basis_hwk[:, :, k], cmap="gray")
        plt.axis("off")
        plt.title(f"e{k+1}")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

# ---------------------------
# Main experiment
# ---------------------------
def main(mat_path: str = "assignment1bases.mat", n_examples_plot: int = 6):
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found {mat_path.resolve()}")

    data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    # Extract cell data
    bases_cell = data["bases"]
    stacks_cell = data["stacks"]

    # Three bases: bases{1}, bases{2}, bases{3}
    B1 = _to_np_basis(_from_mat_cell(bases_cell, 0))  # (19,19,4)
    B2 = _to_np_basis(_from_mat_cell(bases_cell, 1))
    B3 = _to_np_basis(_from_mat_cell(bases_cell, 2))
    bases_list = [B1, B2, B3]

    # Two test sets: stacks{1} (general images), stacks{2} (face images)
    S1 = _to_np_stack(_from_mat_cell(stacks_cell, 0))  # (19,19,400)
    S2 = _to_np_stack(_from_mat_cell(stacks_cell, 1))  # (19,19,400)
    stacks_list = [("General (stack 1)", S1), ("Faces (stack 2)", S2)]

    # --- Visualization ---
    show_examples(S1, "Test Set 1: General", n_examples_plot)
    show_examples(S2, "Test Set 2: Faces", n_examples_plot)
    show_basis(B1, "Basis #1 (four elements)")
    show_basis(B2, "Basis #2 (four elements)")
    show_basis(B3, "Basis #3 (four elements)")

    # --- Mean error table ---
    mean_table = np.zeros((2, 3), dtype=np.float64)
    for si, (sname, S) in enumerate(stacks_list):
        for bi, B in enumerate(bases_list):
            mean_err, _ = batch_mean_error(S, B, ensure_orthonormal=True)
            mean_table[si, bi] = mean_err
            print(f"[RESULT] {sname} vs Basis #{bi+1}: mean error = {mean_err:.6f}")

    # Format and print table
    print("\n=== Mean error norms ===")
    header = "TestSet \\ Basis |  #1         #2         #3"
    print(header)
    print("-"*len(header))
    row1 = f"General (1)      |  {mean_table[0,0]:.6f}  {mean_table[0,1]:.6f}  {mean_table[0,2]:.6f}"
    row2 = f"Faces   (2)      |  {mean_table[1,0]:.6f}  {mean_table[1,1]:.6f}  {mean_table[1,2]:.6f}"
    print(row1)
    print(row2)

    plt.show()

if __name__ == "__main__":
    main()

import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Step 1 : Convert list into NumPY arrays
    param = np.asarray(param,dtype=np.float64)
    grad= np.asarray(grad,dtype=np.float64)
    m=np.asarray(m,dtype=np.float64)
    v=np.asarray(v,dtype=np.float64)

    # Step2: Update the first Moment
    m_new=beta1*m +(1-beta1)*grad

    # Step 3 : Update the Second Moment
    v_new = beta2*v + (1-beta2)*(grad**2)

    # Step 4 : Bias Correction
    m_hat = m_new/(1-beta1**t)
    v_hat = v_new/(1-beta2**t)

    # Step 5 : Update the Parameters
    param_new = param -lr*(m_hat/(np.sqrt(v_hat)+eps))
    
    return param_new,m_new,v_new

    
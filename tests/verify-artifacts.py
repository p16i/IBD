import numpy as np 
import pytest


def test():
    arr_keys = ["rankings", "errvar", "coefficients", "residuals_T"]

    actual = np.load(
        # we generate this one
        "./result/pytorch_resnet18_places365/decompose.npy.npz",
    )
     
    
    expected =  dict(
       zip(
            arr_keys,
            np.load(
                # the original IBD
                "./result/pytorch_resnet18_places365/decompose.npy",
                allow_pickle=True
            )
        )
    )
    
    for key in arr_keys:
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            atol=1e-6
        )
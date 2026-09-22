import unittest
import numpy as np
from meridian_demo.synthetic import generate, validate, adstock, SLUGS

class SyntheticDataTests(unittest.TestCase):
    def test_reproducibility_and_panel(self):
        a,t=generate(); b,_=generate()
        self.assertTrue(a.equals(b)); self.assertEqual(len(a),416)
        validate(a)
        self.assertFalse(a.equals(generate(seed=12)[0]))

    def test_accounting_and_roi_truth(self):
        data,truth=generate()
        revenue=truth.baseline+truth.noise+truth[[f'{s}_contribution' for s in SLUGS]].sum(axis=1)
        np.testing.assert_allclose(data.revenue,revenue)
        for s,target in zip(SLUGS,[3,2,1.5]):
            self.assertAlmostEqual(truth[f'{s}_contribution'].sum()/data[f'{s}_spend'].sum(),target)

    def test_adstock_is_causal_and_finite(self):
        impulse=np.zeros(15); impulse[3]=1
        response=adstock(impulse,.5)
        np.testing.assert_array_equal(response[:3],0)
        np.testing.assert_array_equal(response[10:],0)
        self.assertAlmostEqual(response.sum(),1)

    def test_rejects_invalid_panel(self):
        df,_=generate()
        for bad in [df.iloc[1:],df._append(df.iloc[[0]]),df.assign(revenue=np.nan)]:
            with self.assertRaises(ValueError):validate(bad)

if __name__=='__main__':unittest.main()

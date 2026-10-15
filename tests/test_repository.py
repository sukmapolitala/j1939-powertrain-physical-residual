from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from continuous_state_residual import ContinuousStateResidualDetector
from run_primary_strict import metrics, split_forward, CONTEXTS, RESPONSES

class TestResearchPublicationRepository(unittest.TestCase):
    def test_strict_split_is_ordered(self):
        frame = pd.DataFrame({'timestamp': np.arange(100) * .02})
        tr, ca, te = split_forward(frame)
        self.assertEqual((len(tr),len(ca),len(te)), (60,20,20))
        self.assertLess(tr.timestamp.iloc[-1], ca.timestamp.iloc[0])
        self.assertLess(ca.timestamp.iloc[-1], te.timestamp.iloc[0])

    def test_metrics(self):
        out=metrics([0,0,1,1],[0.1,0.2,0.8,0.9],[0,0,1,1])
        self.assertEqual(out['f1'],1.0)
        self.assertEqual((out['tn'],out['fp'],out['fn'],out['tp']),(2,0,0,2))

    def test_no_forbidden_in_predictor_lists(self):
        disallowed={'pgn','can_id','source','destination','label','attack_frame','perturbation_envelope','priority'}
        self.assertFalse(disallowed.intersection(CONTEXTS+RESPONSES))

    def test_repository_data_access_policy(self):
        import subprocess
        restricted=[
            'data/local_raw/normal.log',
            'data/local_raw/labeled_TSC1.csv',
            'data/local_derived/20ms/labeled_TSC1__physical_features_20ms.csv',
            'results/primary_predictions_20ms.csv',
            'models/proposed_detector.joblib',
        ]
        if not (ROOT/'.git').exists():
            contents=(ROOT/'.gitignore').read_text()
            self.assertIn('data/local_raw/',contents)
            self.assertIn('data/local_derived/',contents)
            self.assertIn('models/',contents)
        else:
            for path in restricted:
                result=subprocess.run(['git','check-ignore','-q',path],cwd=ROOT)
                self.assertEqual(result.returncode,0,f'Must be ignored: {path}')

if __name__ == '__main__':
    unittest.main()

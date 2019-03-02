import auto_diff as ad
from .util import NumGradCheck


class TestOpInTrainPhase(NumGradCheck):

    def test_forward(self):
        x = ad.in_train_phase()
        self.assertEqual(0.0, x.forward())
        self.assertEqual(0.0, x.forward({ad.Operation.KEY_TRAINING: False}))
        self.assertEqual(1.0, x.forward({ad.Operation.KEY_TRAINING: True}))

    def test_backward(self):
        x = ad.in_train_phase()
        x.forward()
        x.backward()

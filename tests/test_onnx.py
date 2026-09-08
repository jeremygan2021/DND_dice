import unittest
from unittest.mock import patch
import numpy as np
from dice_onnx_engine import _yolo_decode, _as_probabilities, _text_to_int, OnnxDiceEngine
from dice_detector import DiceDetector, request_detector_mode


class OnnxTests(unittest.TestCase):
    def test_yolox_objectness(self):
        boxes,scores,classes=_yolo_decode(np.array([[[20,20,10,10,.8,.1,.9]]]),.5,['a','b'])
        np.testing.assert_allclose(boxes,[[15,15,25,25]])
        self.assertAlmostEqual(scores[0],.72)
        self.assertEqual(classes[0],1)

    def test_yolov8_transposed(self):
        x=np.zeros((1,6,10),np.float32)
        x[0,:,0]=[20,20,10,10,.1,.9]
        boxes,scores,classes=_yolo_decode(x,.5,['a','b'])
        self.assertEqual(len(boxes),1)
        self.assertAlmostEqual(float(scores[0]),.9,places=6)

    def test_probabilities_not_softmax_twice(self):
        p=np.array([.8,.1,.1],np.float32)
        np.testing.assert_allclose(_as_probabilities(p),p)

    def test_percentile_unsupported_and_d10_zero(self):
        self.assertIsNone(_text_to_int('3','d100'))
        self.assertEqual(_text_to_int('0','d10'),10)

    def test_glyph_requires_matching_type_and_containment(self):
        e=OnnxDiceEngine.__new__(OnnxDiceEngine)
        e.detect_glyphs=lambda _: [dict(bbox=[20,20,40,40],dice_type='d8',confidence=.9),dict(bbox=[110,20,130,40],dice_type='d20',confidence=.99)]
        self.assertFalse(e.detect_glyph(None,[0,0,100,100],'d20')[0])
        self.assertEqual(e.detect_glyph(None,[0,0,100,100],'d8')[1],(20,20,40,40))

    def test_conflicting_readers_abstain(self):
        e=OnnxDiceEngine.__new__(OnnxDiceEngine)
        e.detect_glyph=lambda *a:(True,(1,1,20,20))
        e.read_value=lambda *a:(6,.9,'6')
        e.read_value_cls=lambda *a:(9,.9,'9')
        result=e.read_die(None,[0,0,100,100],'d10')
        self.assertIsNone(result['value'])
        self.assertEqual(result['reason'],'model_disagreement')

    def test_request_mode_restored(self):
        d=DiceDetector()
        previous=d.backend
        token=request_detector_mode.set('cv')
        try:self.assertEqual(d.backend,'cv')
        finally:request_detector_mode.reset(token)
        self.assertEqual(d.backend,previous)

    def test_low_value_score_not_renormalized(self):
        class Input: name='x'
        class Session:
            def get_inputs(self):return [Input()]
            def run(self,*args):return [np.array([[0.,0.,0.,0.,0.,0.,0.,0.,10.]],np.float32)]
        e=OnnxDiceEngine.__new__(OnnxDiceEngine)
        e._ensure=lambda:None
        e.sessions={'dice-value':Session()}
        e.value_classes=[str(i) for i in range(9)]
        # Globally confident 8 must not be silently remapped into a D6 value.
        v,c,t=e.read_value(np.zeros((100,100,3),np.uint8),[20,20,60,60],'d6')
        self.assertIsNone(v)
        self.assertEqual(t,'8')

if __name__ == '__main__':unittest.main()

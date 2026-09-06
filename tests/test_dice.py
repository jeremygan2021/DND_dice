import unittest
from unittest.mock import patch
import numpy as np
from dice_engine import text_to_value, FaceOCR, DiceSession, Track, _roi_fingerprint
from dice_detector import suppress


class DiceTests(unittest.TestCase):
    def test_values(self):
        for typ, good, bad in [('d4','4','5'),('d6','6','7'),('d8','8','9'),('d12','12','13'),('d20','20','21')]:
            self.assertEqual(text_to_value(good, typ)[0], int(good))
            self.assertIsNone(text_to_value(bad, typ)[0])
        self.assertEqual(text_to_value('0','d10')[0],10)
        self.assertEqual(text_to_value('00','d100')[0],0)
        self.assertIsNone(text_to_value('12','d100')[0])
        self.assertIsNone(text_to_value('0')[0])
        self.assertIsNone(text_to_value('roll 12')[0])
        self.assertIsNone(text_to_value('99')[0])
        self.assertIsNone(text_to_value('06')[0])

    def test_suppression(self):
        ds = [dict(bbox=b,confidence=c) for b,c in [([0,0,100,100],.9),([5,5,95,95],.8),([120,0,200,100],.7)]]
        self.assertEqual(len(suppress(ds)),2)

    @patch('dice_engine._ocr_candidates', side_effect=lambda im: [im,im])
    def test_ocr_consensus_and_ambiguity(self, _):
        o=FaceOCR()
        im=np.zeros((60,60,3),np.uint8)
        box=[[20,20],[40,20],[40,40],[20,40]]
        o._reader=lambda _: ([[box,'12',.95]],None)
        self.assertEqual(o.read_crop(im,'d20')[0],12)
        self.assertIsNone(o.read_crop(im,'d6')[0])
        o._reader=lambda _: ([[box,'12',.95],[box,'8',.99]],None)
        self.assertIsNone(o.read_crop(im,'d20')[0])

    def test_legacy_ctc_confidence(self):
        from ocr_compat import restore_legacy_confidence
        out=restore_legacy_confidence([('4',.49),('20',.64),('',1e-50)])
        self.assertAlmostEqual(out[0][1],.98)
        self.assertAlmostEqual(out[1][1],.96)
        self.assertEqual(out[2][1],0)

    def test_d4_apex_consensus(self):
        o=FaceOCR()
        im=np.zeros((100,100,3),np.uint8)
        a=[[35,40],[45,40],[45,50],[35,50]]
        b=[[55,40],[65,40],[65,50],[55,50]]
        with patch('dice_engine._ocr_candidates',return_value=[im,im]):
            o._reader=lambda _: ([[a,'4',.95],[b,'4',.94]],None)
            self.assertEqual(o.read_crop(im,'d4')[0],4)
            o._reader=lambda _: ([[a,'4',.95],[b,'3',.94]],None)
            self.assertIsNone(o.read_crop(im,'d4')[0])

    def test_motion_clears_value(self):
        s=DiceSession('test',FaceOCR())
        im=np.zeros((200,200,3),np.uint8)
        tr=Track(1,[10,10,70,70],value=6,settled=True,center_px=(40,40),total_frames=5)
        tr._prev_fp=_roi_fingerprint(im,tr.box)
        s.tracks[1]=tr
        s._next_tid=2
        s._update_tracks([[25,10,85,70]],im,0)
        self.assertIsNone(tr.value)
        self.assertFalse(tr.settled)

    def test_periodic_detection_and_hidden_misses(self):
        s=DiceSession('test',FaceOCR())
        im=np.zeros((200,200,3),np.uint8)
        tr=Track(1,[10,10,70,70],value=6,settled=True,center_px=(40,40),total_frames=5)
        s.tracks[1]=tr
        s.prev_boxes=[tr.box]
        s._prev_full=(200,200)
        s._last_detection=0
        with patch('dice_engine.detector.detect',return_value=[]) as detect:
            result=s.process_frame(im)
            detect.assert_called_once()
            self.assertEqual(result['num_dice'],0)
            self.assertEqual(result['total_value'],0)

    def test_ocr_exception_releases_pending(self):
        class Broken:
            def read_crop(self,*args): raise RuntimeError('test')
            def submit(self,fn): fn()
        s=DiceSession('test',Broken())
        t=Track(1,[10,10,70,70],settled=True)
        s.tracks[1]=t
        s._maybe_ocr(t,np.zeros((100,100,3),np.uint8),10)
        self.assertFalse(t.ocr_pending)
        self.assertEqual(t.ocr_fails,1)

if __name__ == '__main__': unittest.main()

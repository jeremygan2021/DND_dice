import unittest
from unittest.mock import patch
import numpy as np
from dice_engine import text_to_value, FaceOCR, DiceSession, Track, _roi_fingerprint, bucket_unknown_by_position
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

    def test_bucket_smaller_cluster_is_d100(self):
        # 3 unknown on the right, 1 unknown on the left → left becomes d100.
        ds = [
            dict(bbox=[10, 50, 80, 130], dice_type='unknown', confidence=.8),
            dict(bbox=[200, 50, 270, 130], dice_type='unknown', confidence=.8),
            dict(bbox=[290, 50, 360, 130], dice_type='unknown', confidence=.8),
            dict(bbox=[400, 50, 470, 130], dice_type='unknown', confidence=.8),
        ]
        bucket_unknown_by_position(ds)
        types = [d['dice_type'] for d in ds]
        self.assertEqual(types[0], 'd100')
        self.assertEqual(types[1:], ['d10', 'd10', 'd10'])

    def test_bucket_tie_prefers_left(self):
        ds = [
            dict(bbox=[10, 50, 80, 130], dice_type='unknown', confidence=.8),
            dict(bbox=[400, 50, 470, 130], dice_type='unknown', confidence=.8),
        ]
        bucket_unknown_by_position(ds)
        # Tie (1 vs 1) → left becomes d100.
        self.assertEqual(ds[0]['dice_type'], 'd100')
        self.assertEqual(ds[1]['dice_type'], 'd10')

    def test_bucket_skips_when_already_typed(self):
        # A typed d6 should never be re-bucketed.
        ds = [
            dict(bbox=[10, 50, 80, 130], dice_type='unknown', confidence=.8),
            dict(bbox=[400, 50, 470, 130], dice_type='d6', confidence=.8),
        ]
        bucket_unknown_by_position(ds)
        self.assertEqual(ds[0]['dice_type'], 'unknown')
        self.assertEqual(ds[1]['dice_type'], 'd6')

    def test_bucket_no_split_when_all_clustered(self):
        # All bbox centres within a few px → can't split.
        ds = [
            dict(bbox=[10, 50, 30, 70], dice_type='unknown', confidence=.8),
            dict(bbox=[11, 50, 31, 70], dice_type='unknown', confidence=.8),
        ]
        bucket_unknown_by_position(ds)
        self.assertEqual(ds[0]['dice_type'], 'unknown')
        self.assertEqual(ds[1]['dice_type'], 'unknown')

    @patch('dice_engine._ocr_candidates', side_effect=lambda im: [im])
    def test_d100_two_digit_text(self, _):
        o = FaceOCR()
        im = np.zeros((100, 100, 3), np.uint8)
        box = [[10, 10], [80, 10], [80, 80], [10, 80]]
        o._reader = lambda _: ([[box, '70', .95]], None)
        self.assertEqual(o.read_crop(im, 'd100')[0], 70)

    @patch('dice_engine._ocr_candidates', side_effect=lambda im: [im])
    def test_d100_pair_single_digits(self, _):
        # DBNet splits "70" into two boxes that we should stitch back together.
        o = FaceOCR()
        im = np.zeros((100, 100, 3), np.uint8)
        a = [[10, 20], [35, 20], [35, 70], [10, 70]]   # "7"
        b = [[45, 20], [70, 20], [70, 70], [45, 70]]   # "0"
        o._reader = lambda _: ([[a, '7', .95], [b, '0', .94]], None)
        v, c, t = o.read_crop(im, 'd100')
        self.assertEqual(v, 70)
        self.assertEqual(t, '70')

    @patch('dice_engine._ocr_candidates', side_effect=lambda im: [im])
    def test_d100_refuses_wrong_pair(self, _):
        # Digit pair that doesn't form a multiple of ten → reject.
        o = FaceOCR()
        im = np.zeros((100, 100, 3), np.uint8)
        a = [[10, 20], [35, 20], [35, 70], [10, 70]]
        b = [[45, 20], [70, 20], [70, 70], [45, 70]]
        o._reader = lambda _: ([[a, '3', .95], [b, '7', .94]], None)
        self.assertIsNone(o.read_crop(im, 'd100')[0])

    @patch('dice_engine._ocr_candidates', side_effect=lambda im: [im])
    def test_d100_refuses_stacked_or_far_apart(self, _):
        # Two boxes stacked vertically or with too-large horizontal gap → don't pair.
        o = FaceOCR()
        im = np.zeros((100, 100, 3), np.uint8)
        a = [[10, 10], [35, 10], [35, 30], [10, 30]]
        b = [[200, 80], [240, 80], [240, 100], [200, 100]]
        o._reader = lambda _: ([[a, '7', .95], [b, '0', .94]], None)
        self.assertIsNone(o.read_crop(im, 'd100')[0])

    def test_session_uses_bucketed_type(self):
        # When expected_type=unknown, position-based split overrides detections.
        s = DiceSession('test-bucket', FaceOCR())
        im = np.zeros((200, 400, 3), np.uint8)
        boxes = [[10, 50, 80, 130], [200, 50, 270, 130]]
        detections = [
            dict(bbox=boxes[0], dice_type='unknown', confidence=.8),
            dict(bbox=boxes[1], dice_type='unknown', confidence=.8),
        ]
        with patch('dice_engine.detector.detect', return_value=detections):
            s.process_frame(im, dice_type='unknown')
        types = sorted(t.dice_type for t in s.tracks.values())
        self.assertEqual(types, ['d10', 'd100'])

    def test_session_pinned_type_ignores_bucket(self):
        # When the user explicitly picks a dice type, position-based split is off.
        s = DiceSession('test-pinned', FaceOCR())
        im = np.zeros((200, 400, 3), np.uint8)
        boxes = [[10, 50, 80, 130], [200, 50, 270, 130]]
        detections = [
            dict(bbox=boxes[0], dice_type='unknown', confidence=.8),
            dict(bbox=boxes[1], dice_type='unknown', confidence=.8),
        ]
        with patch('dice_engine.detector.detect', return_value=detections):
            s.process_frame(im, dice_type='d6')
        for t in s.tracks.values():
            self.assertEqual(t.dice_type, 'd6')


if __name__ == '__main__': unittest.main()

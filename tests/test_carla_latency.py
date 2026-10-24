"""GPU-delayed callbacks must retain their own snapshot/pose, without frame dropping."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from src.carla_capture import CarlaCapture, CaptureSettings, FrameSynchronizer


def transform(x):
    return SimpleNamespace(location=SimpleNamespace(x=x,y=0,z=0), rotation=SimpleNamespace(pitch=0,yaw=0,roll=0))


class DelayedCameraTests(unittest.TestCase):
    def capture(self, skip=None, bad_time=False):
        capture=CarlaCapture(CaptureSettings(timeout=2))
        capture.sync=FrameSynchronizer(4)
        capture.vehicle=SimpleNamespace(id=42)
        state={'frame':0}
        raw=bytes([1,2,3,255])*(640*480)
        def tick(timeout):
            state['frame']+=1
            old=state['frame']-2
            if old>0 and old!=skip:
                for name in ('rgb','depth'):
                    stamp=old*0.1+(0.5 if bad_time and name=='depth' else 0)
                    capture.sync.put(name,SimpleNamespace(frame=old,timestamp=stamp,raw_data=raw,transform=transform(old)))
            return state['frame']
        def snapshot():
            frame=state['frame']
            vehicle=SimpleNamespace(get_transform=lambda:transform(frame))
            return SimpleNamespace(frame=frame,timestamp=SimpleNamespace(elapsed_seconds=frame*0.1),find=lambda actor:vehicle)
        capture.world=SimpleNamespace(tick=tick,get_snapshot=snapshot)
        return capture,state

    def test_delayed_pair_uses_historical_pose_and_preserves_consecutive_frames(self):
        capture,state=self.capture()
        first=capture.next_frame()
        self.assertEqual(state['frame'],3)
        self.assertEqual(first['frame_id'],1)
        self.assertEqual(first['ego_transform']['location']['x'],1)
        self.assertAlmostEqual(first['simulation_timestamp'],0.1)
        second=capture.next_frame()
        self.assertEqual(second['frame_id'],2)
        self.assertEqual(second['ego_transform']['location']['x'],2)
        self.assertLessEqual(len(capture.snapshots),capture.sync.capacity)

    def test_only_explicit_unrecorded_warmup_may_skip_startup_gaps(self):
        capture,state=self.capture(skip=2)
        self.assertEqual(capture.next_frame(warmup=True)['frame_id'],1)
        self.assertIsNone(capture.expected_frame)
        self.assertEqual(capture.next_frame(warmup=True)['frame_id'],3)
        self.assertEqual(capture.next_frame()['frame_id'],4)
        self.assertEqual(capture.expected_frame,5)

    def test_cleanup_unregisters_autopilot_before_destroying_vehicle(self):
        capture=CarlaCapture(CaptureSettings())
        events=[]
        capture.vehicle=MagicMock()
        capture.vehicle.type_id='vehicle.test'
        capture.vehicle.set_autopilot.side_effect=lambda enabled,port:events.append(('autopilot',enabled,port))
        capture.vehicle.destroy.side_effect=lambda:events.append(('destroy',))
        capture.actors=[capture.vehicle]
        capture.traffic_manager=MagicMock()
        capture.__exit__(None,None,None)
        self.assertEqual(events,[('autopilot',False,8001),('destroy',)])

    def test_missing_middle_frame_cannot_be_silently_skipped(self):
        capture,state=self.capture(skip=2)
        self.assertEqual(capture.next_frame()['frame_id'],1)
        with self.assertRaisesRegex(TimeoutError,'bounded'):
            capture.next_frame()
        self.assertEqual(capture.expected_frame,2)
        self.assertLessEqual(len(capture.snapshots),4)

    def test_mismatched_sensor_timestamps_and_duplicate_callbacks_fail(self):
        capture,_=self.capture(bad_time=True)
        with self.assertRaisesRegex(RuntimeError,'timestamps'):
            capture.next_frame()
        sync=FrameSynchronizer()
        sample=SimpleNamespace(frame=1)
        sync.put('rgb',sample);sync.put('rgb',sample)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            sync.poll_pair()


if __name__=='__main__':
    unittest.main()

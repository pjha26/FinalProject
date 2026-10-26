"""Supervised metric loss numerical/gradient and teacher-isolation checks."""
import unittest
import torch
from src.metric_training import metric_supervision_loss


class MetricTrainingTests(unittest.TestCase):
    def fixture(self):
        inverse=torch.full((1,1,4,4),2.,requires_grad=True)
        gt=torch.full_like(inverse,5.)
        mask=torch.ones_like(inverse,dtype=torch.bool)
        return inverse,gt,mask,mask.clone(),inverse.detach().clone().requires_grad_()

    def test_exact_depth_has_zero_loss(self):
        loss,_=metric_supervision_loss(*self.fixture(),scale=10)
        self.assertEqual(float(loss),0.)

    def test_gradient_updates_prediction_not_teacher(self):
        inverse,gt,mask,obj,teacher=self.fixture()
        gt=gt*2
        loss,_=metric_supervision_loss(inverse,gt,mask,obj,teacher,10)
        loss.backward()
        self.assertTrue(torch.isfinite(inverse.grad).all())
        self.assertGreater(float(inverse.grad.abs().sum()),0)
        self.assertIsNone(teacher.grad)

    def test_object_region_adds_explicit_weight(self):
        inverse,gt,mask,obj,teacher=self.fixture()
        gt=gt.clone();gt[:,:,:1,:]=10
        only_object=torch.zeros_like(obj);only_object[:,:,:1,:]=True
        without,_=metric_supervision_loss(inverse,gt,mask,torch.zeros_like(obj),teacher,10)
        with_object,_=metric_supervision_loss(inverse,gt,mask,only_object,teacher,10)
        self.assertGreater(float(with_object),float(without))

    def test_invalid_supervision_rejected(self):
        values=list(self.fixture())
        for scale in [0,float("nan")]:
            with self.assertRaises(ValueError):metric_supervision_loss(*values,scale)
        bad=values.copy();bad[2]=torch.zeros_like(values[2])
        with self.assertRaises(ValueError):metric_supervision_loss(*bad,10)
        bad=values.copy();bad[1]=torch.full_like(values[1],float("nan"))
        with self.assertRaises(ValueError):metric_supervision_loss(*bad,10)
        bad=values.copy();bad[0]=torch.zeros_like(values[0])
        with self.assertRaises(ValueError):metric_supervision_loss(*bad,10)





class TeacherReplayTests(unittest.TestCase):
    def test_real_teacher_loss_is_exact_and_detached(self):
        from src.metric_training import teacher_consistency_loss
        inverse=torch.full((1,1,4,4),4.,requires_grad=True)
        teacher=torch.full_like(inverse,2.,requires_grad=True)
        loss=teacher_consistency_loss(inverse,teacher)
        self.assertAlmostEqual(float(loss),float(torch.log(torch.tensor(2.))),places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(inverse.grad).all())
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            teacher_consistency_loss(inverse,teacher*0)



class TrainingBudgetTests(unittest.TestCase):
    def test_unbounded_update_budget_is_rejected_before_outputs(self):
        from tools.train_metric_decoder import run
        from unittest.mock import patch
        with patch("pathlib.Path.mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError,"bounded"):
                run("unused",steps=100000)
            mkdir.assert_not_called()

if __name__=="__main__":unittest.main()

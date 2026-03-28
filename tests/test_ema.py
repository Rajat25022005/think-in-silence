import copy
import pytest
import torch
import torch.nn as nn
from src.training.schedulers import update_ema


def make_model():
    return nn.Sequential(nn.Linear(8, 8))


def test_teacher_params_change_after_update():
    student = make_model()
    teacher = copy.deepcopy(student)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = [p.clone() for p in teacher.parameters()]
    update_ema(student, teacher, momentum=0.99)
    for before_p, after_p in zip(before, teacher.parameters()):
        assert not torch.allclose(before_p, after_p)


def test_higher_momentum_smaller_change():
    student  = make_model()
    teacher1 = copy.deepcopy(student)
    teacher2 = copy.deepcopy(student)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.ones_like(p))
    update_ema(student, teacher1, momentum=0.9)
    update_ema(student, teacher2, momentum=0.999)
    change1 = sum(
        (t - s).abs().sum().item()
        for t, s in zip(teacher1.parameters(), copy.deepcopy(student).parameters())
    )
    change2 = sum(
        (t - s).abs().sum().item()
        for t, s in zip(teacher2.parameters(), copy.deepcopy(student).parameters())
    )
    assert change2 < change1


def test_teacher_requires_grad_false():
    student = make_model()
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    update_ema(student, teacher, momentum=0.99)
    for p in teacher.parameters():
        assert not p.requires_grad
